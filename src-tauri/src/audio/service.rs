//! Application-owned microphone capture lifecycle.
//!
//! `CaptureService` owns the native stream and only exposes compact metadata.
//! Raw PCM stays inside the pre-allocated ingress and its native dispatcher.

use super::{
    capture_point_now, AsrQueueMetrics, CaptureClock, CaptureFailureCode, CaptureGap,
    CaptureGapReason, CaptureLifecycle, CapturePoint, CaptureStatus, CpalInput, CpalInputFailure,
    DispatcherRuntime, InputDeviceList, MacOsCaptureEvent, MacOsInputDevice, MicrophonePermission,
    NativeCaptureRuntime, NativeCaptureRuntimeConfig, NativeCaptureRuntimeSnapshot,
    NativeCaptureRuntimeStatus, NativeInferenceEngines, OwnedOutcomeLease,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_PENDING_GAPS: usize = 16;
const SUPPORTED_CAPTURE_SAMPLE_RATES: [u32; 2] = [16_000, 48_000];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureIssueCode {
    PermissionDenied,
    PermissionRestricted,
    NoInputDevice,
    InputDeviceUnavailable,
    StreamStartFailed,
    CaptureQueueOverrun,
    CaptureQueueClosed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureIssue {
    pub code: CaptureIssueCode,
    pub device_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureMeter {
    pub rms_dbfs: f32,
    pub peak_dbfs: f32,
    pub clipping: bool,
    pub dropped_packets: u64,
}

/// A serializable lifecycle view of the native bridge. It contains bounded
/// queue state only; PCM and transcript text remain native-only.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureBridgeStatus {
    Parked,
    Armed,
    Closing,
    Drained,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureBridgeProjection {
    pub status: CaptureBridgeStatus,
    pub armed: bool,
    pub shutdown_requested: bool,
    pub worker_finished: bool,
    pub metrics: AsrQueueMetrics,
}

impl CaptureBridgeProjection {
    fn from_snapshot(snapshot: &NativeCaptureRuntimeSnapshot) -> Self {
        let status = match snapshot.status {
            NativeCaptureRuntimeStatus::Parked => CaptureBridgeStatus::Parked,
            NativeCaptureRuntimeStatus::Armed => CaptureBridgeStatus::Armed,
            NativeCaptureRuntimeStatus::Closing => CaptureBridgeStatus::Closing,
            NativeCaptureRuntimeStatus::Drained => CaptureBridgeStatus::Drained,
        };
        Self {
            status,
            armed: snapshot.armed,
            shutdown_requested: snapshot.shutdown_requested,
            worker_finished: snapshot.worker_finished,
            metrics: snapshot.metrics.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureProjection {
    pub revision: u64,
    pub status: CaptureStatus,
    pub permission: MicrophonePermission,
    pub selected_device: Option<MacOsInputDevice>,
    pub devices: Vec<MacOsInputDevice>,
    pub meter: Option<CaptureMeter>,
    pub bridge: Option<CaptureBridgeProjection>,
    pub last_issue: Option<CaptureIssue>,
}

#[derive(Clone, Debug)]
pub struct CapturePreparation {
    pub anchor: CapturePoint,
    pub device: MacOsInputDevice,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Metadata for a CPAL stream that has been played but is not published as
/// recording until [`CaptureService::arm_after_commit`] succeeds.
#[derive(Clone, Debug)]
pub struct CaptureStart {
    pub anchor: CapturePoint,
    pub device: MacOsInputDevice,
    pub sample_rate: u32,
    pub channels: u16,
    pub runtime: DispatcherRuntime,
}

/// A retryable claim over the oldest capture discontinuity awaiting durable
/// storage. The gap stays queued until its token is committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureGapLease {
    token: u64,
    gap: CaptureGap,
}

impl CaptureGapLease {
    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn gap(&self) -> &CaptureGap {
        &self.gap
    }
}

pub struct CaptureService {
    lifecycle: CaptureLifecycle,
    permission: MicrophonePermission,
    devices: Vec<MacOsInputDevice>,
    selected_device_uid: Option<String>,
    meter: Option<CaptureMeter>,
    bridge: Option<CaptureBridgeProjection>,
    last_issue: Option<CaptureIssue>,
    revision: u64,
    observed_dropped_packets: u64,
    pending_gaps: VecDeque<CaptureGap>,
    active_gap_lease: Option<CaptureGapLease>,
    next_gap_lease_token: u64,
    active_queue_overrun_gap: Option<CaptureGap>,
    input: Option<CpalInput>,
    prepared_capture: Option<CapturePreparation>,
    runtime: Option<NativeCaptureRuntime>,
    // A pre-arm worker that was explicitly shut down after startup failed.
    // It must be moved out and joined after the service mutex is released.
    prearm_runtimes_for_join: VecDeque<NativeCaptureRuntime>,
}

impl Default for CaptureService {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureService {
    pub fn new() -> Self {
        Self {
            lifecycle: CaptureLifecycle::new(),
            permission: MicrophonePermission::NotDetermined,
            devices: Vec::new(),
            selected_device_uid: None,
            meter: None,
            bridge: None,
            last_issue: None,
            revision: 0,
            observed_dropped_packets: 0,
            pending_gaps: VecDeque::with_capacity(MAX_PENDING_GAPS),
            active_gap_lease: None,
            next_gap_lease_token: 1,
            active_queue_overrun_gap: None,
            input: None,
            prepared_capture: None,
            runtime: None,
            prearm_runtimes_for_join: VecDeque::new(),
        }
    }

    pub fn projection(&mut self) -> CaptureProjection {
        self.refresh();
        CaptureProjection {
            revision: self.revision,
            status: self.lifecycle.status(),
            permission: self.permission,
            selected_device: self.selected_device_for_projection(),
            devices: self.devices.clone(),
            meter: self.meter.clone(),
            bridge: self.bridge.clone(),
            last_issue: self.last_issue.clone(),
        }
    }

    /// Backward-compatible destructive delivery for callers that do not claim
    /// a gap lease. New durable callers must use the begin/commit/abort API so
    /// a failed SQLite write can retry the same discontinuity.
    pub fn take_pending_gaps(&mut self) -> Vec<CaptureGap> {
        if self.active_gap_lease.is_some() {
            return Vec::new();
        }
        self.pending_gaps.drain(..).collect()
    }

    /// Claim the oldest physical capture gap without removing it. The caller
    /// must commit after durable storage or abort to retry the same gap.
    pub fn begin_pending_gap(&mut self) -> Result<Option<CaptureGapLease>, String> {
        if self.active_gap_lease.is_some() {
            return Err("a capture gap delivery is already active".to_owned());
        }
        let Some(gap) = self.pending_gaps.front().cloned() else {
            return Ok(None);
        };
        let token = self.next_gap_lease_token;
        self.next_gap_lease_token = self.next_gap_lease_token.checked_add(1).unwrap_or(1);
        let lease = CaptureGapLease { token, gap };
        self.active_gap_lease = Some(lease.clone());
        Ok(Some(lease))
    }

    pub fn commit_pending_gap(&mut self, token: u64) -> Result<(), String> {
        let lease = self.require_active_gap_lease(token)?;
        if self.pending_gaps.front() != Some(&lease.gap) {
            return Err("the active capture gap no longer matches the pending head".to_owned());
        }
        self.pending_gaps.pop_front();
        self.active_gap_lease = None;
        Ok(())
    }

    pub fn abort_pending_gap(&mut self, token: u64) -> Result<(), String> {
        self.require_active_gap_lease(token)?;
        self.active_gap_lease = None;
        Ok(())
    }

    /// Claim one native inference outcome while retaining the service lock
    /// only for the short dispatcher operation. Persistence must happen after
    /// the caller releases that lock.
    pub fn begin_native_outcome(&self) -> Result<Option<OwnedOutcomeLease>, String> {
        let Some(runtime) = self.runtime.as_ref() else {
            return Ok(None);
        };
        runtime
            .begin_owned_outcome()
            .map_err(|error| format!("could not claim native inference outcome: {error}"))
    }

    pub fn commit_native_outcome(&self, token: u64) -> Result<(), String> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "no native capture runtime is active".to_owned())?;
        runtime
            .commit_owned_outcome(token)
            .map_err(|error| format!("could not commit native inference outcome: {error}"))
    }

    pub fn abort_native_outcome(&self, token: u64) -> Result<(), String> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "no native capture runtime is active".to_owned())?;
        runtime
            .abort_owned_outcome(token)
            .map_err(|error| format!("could not abort native inference outcome: {error}"))
    }

    pub fn runtime_context(&self) -> Result<Option<DispatcherRuntime>, String> {
        self.runtime
            .as_ref()
            .map(|runtime| {
                runtime
                    .runtime()
                    .map_err(|error| format!("could not inspect native capture runtime: {error}"))
            })
            .transpose()
    }

    /// Remove a runtime only after every owned outcome has been durably
    /// handled. The caller must join it after releasing the service mutex.
    pub fn take_drained_native_runtime(&mut self) -> Result<Option<NativeCaptureRuntime>, String> {
        let Some(runtime) = self.runtime.as_ref() else {
            return Ok(None);
        };
        if !runtime
            .is_drained()
            .map_err(|error| format!("could not inspect native capture runtime: {error}"))?
        {
            return Ok(None);
        }
        self.bridge = None;
        self.touch();
        Ok(self.runtime.take())
    }

    /// Move one explicitly aborted pre-arm runtime to the caller.
    ///
    /// The caller must release the `CaptureService` mutex, then invoke
    /// [`NativeCaptureRuntime::join_after_abort`] before starting another
    /// capture. This is intentionally separate from
    /// [`Self::take_drained_native_runtime`], whose runtimes may still have
    /// durable inference outcomes to flush.
    pub fn take_prearm_runtime_for_join(&mut self) -> Option<NativeCaptureRuntime> {
        let runtime = self.prearm_runtimes_for_join.pop_front();
        if runtime.is_some() {
            self.touch();
        }
        runtime
    }

    pub fn select_input_device(&mut self, device_uid: String) -> Result<CaptureProjection, String> {
        self.refresh();
        if self.input.is_some()
            || self.runtime.is_some()
            || !self.prearm_runtimes_for_join.is_empty()
        {
            return Err("stop recording before selecting a different microphone".to_owned());
        }
        if !self.devices.iter().any(|device| device.uid() == device_uid) {
            return Err("the selected microphone is unavailable".to_owned());
        }
        if self.selected_device_uid.as_deref() != Some(device_uid.as_str()) {
            self.selected_device_uid = Some(device_uid);
            self.touch();
        }
        Ok(self.projection())
    }

    /// Prepare permission, device, config, and a paused CPAL stream. This
    /// does not begin ingesting audio or publish a Recording lifecycle state.
    pub fn prepare(&mut self) -> Result<CapturePreparation, String> {
        self.refresh();
        self.flush_queue_overrun_gap();
        if self.input.is_some()
            || self.prepared_capture.is_some()
            || self.runtime.is_some()
            || !self.prearm_runtimes_for_join.is_empty()
        {
            return Err("microphone capture is already preparing or active".to_owned());
        }
        if self.active_gap_lease.is_some() || !self.pending_gaps.is_empty() {
            return Err(
                "pending capture gaps must be durably handled before recording restarts".to_owned(),
            );
        }
        if self.lifecycle.status() == CaptureStatus::Recording {
            return Err("microphone capture is already active".to_owned());
        }

        let requested_at = capture_point_now();
        self.lifecycle
            .begin_permission_resolution(requested_at.clone())
            .map_err(|error| error.to_string())?;
        self.last_issue = None;
        self.meter = None;
        self.bridge = None;
        self.observed_dropped_packets = 0;
        self.active_queue_overrun_gap = None;
        self.touch();

        self.permission = CpalInput::request_permission().inspect_err(|_| {
            let _ = self.lifecycle.fail_with_code(
                requested_at.clone(),
                CaptureFailureCode::External,
                "could not request microphone permission",
            );
            self.last_issue = Some(CaptureIssue {
                code: CaptureIssueCode::StreamStartFailed,
                device_name: None,
            });
            self.touch();
        })?;
        if self.permission == MicrophonePermission::Denied {
            self.fail_permission(requested_at, CaptureFailureCode::PermissionDenied);
            return Err("microphone permission is denied".to_owned());
        }
        if self.permission == MicrophonePermission::Restricted {
            self.fail_permission(requested_at, CaptureFailureCode::PermissionRestricted);
            return Err("microphone permission is restricted by macOS".to_owned());
        }

        // macOS can reveal a device only after permission is granted. Refresh
        // once more at the permission boundary before we validate and open it.
        self.refresh_input_device_directory();

        if self.devices.is_empty() {
            self.fail(
                requested_at,
                CaptureFailureCode::NoInputDevice,
                CaptureIssueCode::NoInputDevice,
                None,
                "no microphone is available",
            );
            return Err("no microphone is available".to_owned());
        }

        let selected_device_uid = self.selected_device_uid.as_deref();
        let (mut input, anchor) = match CpalInput::prepare(selected_device_uid) {
            Ok(prepared) => prepared,
            Err(error) => {
                let code = if error.contains("permission") {
                    CaptureFailureCode::PermissionDenied
                } else if error.contains("no default microphone") || error.contains("unavailable") {
                    CaptureFailureCode::NoInputDevice
                } else {
                    CaptureFailureCode::StreamStartFailed
                };
                let issue = match code {
                    CaptureFailureCode::PermissionDenied => CaptureIssueCode::PermissionDenied,
                    CaptureFailureCode::NoInputDevice => CaptureIssueCode::NoInputDevice,
                    _ => CaptureIssueCode::StreamStartFailed,
                };
                self.fail(
                    requested_at,
                    code,
                    issue,
                    None,
                    "could not start microphone capture",
                );
                return Err(error);
            }
        };

        let device = input.device().clone();
        let sample_rate = input.sample_rate();
        let channels = input.channels();
        if let Err(error) = preflight_input_sample_rate(sample_rate) {
            input.stop();
            self.fail(
                requested_at,
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
                Some(device.name().to_owned()),
                "unsupported microphone input sample rate",
            );
            return Err(error);
        }

        let prepared = CapturePreparation {
            anchor,
            device,
            sample_rate,
            channels,
        };
        self.permission = MicrophonePermission::Granted;
        self.input = Some(input);
        self.prepared_capture = Some(prepared.clone());
        self.touch();

        Ok(prepared)
    }

    /// Construct a parked dispatcher, then hand off the already prepared CPAL
    /// stream. The caller must durably record the session/capture start before
    /// calling [`Self::arm_after_commit`].
    pub fn activate_with_runtime(
        &mut self,
        dispatcher_runtime: DispatcherRuntime,
    ) -> Result<CaptureStart, String> {
        self.activate_with_runtime_inner(
            dispatcher_runtime,
            NativeCaptureRuntimeConfig::default(),
            None,
        )
    }

    /// Activate prepared microphone capture with native-only local VAD and
    /// ASR engines chosen before microphone access. The engines are consumed
    /// by the runtime and never cross a Tauri command boundary.
    pub fn activate_with_runtime_and_engines(
        &mut self,
        dispatcher_runtime: DispatcherRuntime,
        runtime_config: NativeCaptureRuntimeConfig,
        engines: NativeInferenceEngines,
    ) -> Result<CaptureStart, String> {
        self.activate_with_runtime_inner(dispatcher_runtime, runtime_config, Some(engines))
    }

    fn activate_with_runtime_inner(
        &mut self,
        dispatcher_runtime: DispatcherRuntime,
        runtime_config: NativeCaptureRuntimeConfig,
        engines: Option<NativeInferenceEngines>,
    ) -> Result<CaptureStart, String> {
        let prepared = self
            .prepared_capture
            .clone()
            .ok_or_else(|| "microphone capture has not been prepared".to_owned())?;
        if self.runtime.is_some() {
            return Err("native capture runtime is already active".to_owned());
        }
        if dispatcher_runtime.capture_anchor != prepared.anchor {
            self.release_prepared_input();
            self.fail(
                capture_point_now(),
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
                Some(prepared.device.name().to_owned()),
                "dispatcher runtime capture anchor does not match prepared input",
            );
            return Err(
                "dispatcher runtime capture anchor does not match prepared input".to_owned(),
            );
        }

        let clock = match CaptureClock::new(prepared.anchor.clone(), prepared.sample_rate) {
            Ok(clock) => clock,
            Err(error) => {
                self.release_prepared_input();
                self.fail(
                    capture_point_now(),
                    CaptureFailureCode::StreamStartFailed,
                    CaptureIssueCode::StreamStartFailed,
                    Some(prepared.device.name().to_owned()),
                    "could not create the native capture clock",
                );
                return Err(error);
            }
        };
        let ingress = self
            .input
            .as_ref()
            .ok_or_else(|| "prepared microphone input is unavailable".to_owned())?
            .ingress();
        let runtime_result = match engines {
            Some(engines) => NativeCaptureRuntime::new_with_engines(
                ingress,
                dispatcher_runtime.clone(),
                clock,
                runtime_config,
                engines,
            ),
            None => NativeCaptureRuntime::new(
                ingress,
                dispatcher_runtime.clone(),
                clock,
                runtime_config,
            ),
        };
        let native_runtime = match runtime_result {
            Ok(runtime) => runtime,
            Err(error) => {
                self.release_prepared_input();
                self.fail(
                    capture_point_now(),
                    CaptureFailureCode::StreamStartFailed,
                    CaptureIssueCode::StreamStartFailed,
                    Some(prepared.device.name().to_owned()),
                    "could not prepare native microphone runtime",
                );
                return Err(format!(
                    "could not prepare native microphone runtime: {error}"
                ));
            }
        };

        let activation = self
            .input
            .as_mut()
            .expect("prepared microphone input exists while activating")
            .activate();
        if let Err(error) = activation {
            self.release_prepared_input();
            // The caller can hold the service mutex here. Signal the parked
            // worker, retain its ownership, and let AppState join it after
            // releasing this mutex. Dropping the handle here could detach it.
            let cleanup_error = native_runtime
                .abort_before_arm()
                .err()
                .map(|cleanup| format!("could not abort parked native runtime: {cleanup}"));
            if cleanup_error.is_some() {
                let _ = native_runtime.request_shutdown();
            }
            self.retain_prearm_runtime_for_join(native_runtime);
            self.fail(
                capture_point_now(),
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
                Some(prepared.device.name().to_owned()),
                "could not activate microphone capture",
            );
            return Err(match cleanup_error {
                Some(cleanup_error) => {
                    format!("could not activate microphone capture: {error}; {cleanup_error}")
                }
                None => format!("could not activate microphone capture: {error}"),
            });
        }

        self.runtime = Some(native_runtime);
        self.touch();
        Ok(CaptureStart {
            anchor: prepared.anchor,
            device: prepared.device,
            sample_rate: prepared.sample_rate,
            channels: prepared.channels,
            runtime: dispatcher_runtime,
        })
    }

    /// Publish Recording and permit the native dispatcher to consume ingress
    /// only after the caller has committed the session and capture audit rows.
    pub fn arm_after_commit(&mut self) -> Result<CaptureStart, String> {
        let prepared = self
            .prepared_capture
            .clone()
            .ok_or_else(|| "microphone capture has not been prepared".to_owned())?;
        if self.lifecycle.status() != CaptureStatus::AwaitingPermission {
            return Err("microphone capture is not awaiting staged activation".to_owned());
        }
        if self.input.is_none() {
            return Err("prepared microphone input is unavailable".to_owned());
        }
        let dispatcher_runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "native capture runtime has not been activated".to_owned())?
            .runtime()
            .map_err(|error| format!("could not inspect native capture runtime: {error}"))?;
        // The CPAL stream is already playing while the capture-start bundle is
        // made durable. Its cumulative drop counter therefore includes parked
        // pre-commit backpressure, which is not part of this recording. Take
        // the baseline before waking the dispatcher: any loss after this
        // boundary remains visible as an explicit capture gap.
        let dropped_packets_at_arm = self
            .input
            .as_ref()
            .expect("prepared microphone input exists while arming")
            .telemetry()
            .dropped_packets;
        self.set_armed_drop_baseline(dropped_packets_at_arm);
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "native capture runtime has not been activated".to_owned())?;
        if let Err(error) = runtime.arm() {
            let cleanup_error = self
                .stop_input_and_request_runtime_shutdown()
                .err()
                .map(|cleanup| format!("could not stop parked native runtime: {cleanup}"));
            self.fail(
                capture_point_now(),
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
                Some(prepared.device.name().to_owned()),
                "could not arm native microphone capture",
            );
            return Err(match cleanup_error {
                Some(cleanup_error) => {
                    format!("could not arm native capture runtime: {error}; {cleanup_error}")
                }
                None => format!("could not arm native capture runtime: {error}"),
            });
        }
        if let Err(error) = self.lifecycle.apply(MacOsCaptureEvent::CaptureStarted {
            device: prepared.device.clone(),
            at: prepared.anchor.clone(),
        }) {
            let cleanup_error = self
                .stop_input_and_request_runtime_shutdown()
                .err()
                .map(|cleanup| format!("could not stop failed native capture: {cleanup}"));
            self.fail(
                capture_point_now(),
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
                Some(prepared.device.name().to_owned()),
                "could not publish active microphone capture",
            );
            return Err(match cleanup_error {
                Some(cleanup_error) => {
                    format!("could not publish active microphone capture: {error}; {cleanup_error}")
                }
                None => format!("could not publish active microphone capture: {error}"),
            });
        }
        self.permission = MicrophonePermission::Granted;
        self.touch();
        Ok(CaptureStart {
            anchor: prepared.anchor,
            device: prepared.device,
            sample_rate: prepared.sample_rate,
            channels: prepared.channels,
            runtime: dispatcher_runtime,
        })
    }

    /// Cancel a played but unarmed staged start after its audit/session commit
    /// fails. The producer is stopped before the runtime discards ingress, so
    /// this path cannot manufacture an inference outcome or gap.
    pub fn abort_after_failed_commit(&mut self) -> Result<Option<NativeCaptureRuntime>, String> {
        if self.lifecycle.status() == CaptureStatus::Recording {
            return Err("cannot abort a capture runtime after it has been armed".to_owned());
        }

        self.release_prepared_input();
        let Some(runtime) = self.runtime.take() else {
            self.cancel_prepared_lifecycle()?;
            return Ok(None);
        };
        if let Err(error) = runtime.abort_before_arm() {
            self.runtime = Some(runtime);
            return Err(format!(
                "could not abort parked native capture runtime: {error}"
            ));
        }
        self.bridge = None;
        self.meter = None;
        if let Err(error) = self.cancel_prepared_lifecycle() {
            self.retain_prearm_runtime_for_join(runtime);
            return Err(error);
        }
        self.touch();
        Ok(Some(runtime))
    }

    /// Kept for source compatibility while startup is staged. A caller must
    /// supply a session- and segment-fenced dispatcher runtime before CPAL can
    /// be activated safely.
    pub fn start(&mut self) -> Result<CaptureStart, String> {
        Err(
            "microphone capture must use prepare, activate_with_runtime, and arm_after_commit"
                .to_owned(),
        )
    }

    pub fn stop(&mut self) -> Result<bool, String> {
        self.refresh();
        let status = self.lifecycle.status();
        if status == CaptureStatus::Recording {
            self.flush_queue_overrun_gap();
        }
        let input = self.input.take();
        let had_input = input.is_some();
        let device = input
            .as_ref()
            .map(|input| input.device().clone())
            .or_else(|| self.lifecycle.selected_device().cloned());
        if let Some(mut input) = input {
            input.stop();
        }
        self.prepared_capture = None;

        let had_runtime = self.runtime.is_some();
        if !had_input && !had_runtime && status == CaptureStatus::Idle {
            return Ok(false);
        }

        if status == CaptureStatus::AwaitingPermission {
            if let Some(runtime) = self.runtime.take() {
                if let Err(error) = runtime.abort_before_arm() {
                    self.runtime = Some(runtime);
                    return Err(format!(
                        "could not abort parked native capture runtime: {error}"
                    ));
                }
                self.retain_prearm_runtime_for_join(runtime);
            }
            self.bridge = None;
            self.meter = None;
            self.cancel_prepared_lifecycle()?;
            self.touch();
            return Ok(true);
        }

        if let Some(runtime) = self.runtime.as_ref() {
            runtime
                .request_shutdown()
                .map_err(|error| format!("could not stop native capture runtime: {error}"))?;
        }
        if matches!(
            status,
            CaptureStatus::Recording | CaptureStatus::Interrupted | CaptureStatus::Failed
        ) {
            // A failed staged start can own a parked runtime before
            // `CaptureStarted` published a selected device. That runtime must
            // still be drained, but there is no capture lifecycle stop event
            // to emit. A recording or interrupted capture always has one.
            if let Some(device) = device {
                self.lifecycle
                    .apply(MacOsCaptureEvent::CaptureStopped {
                        device,
                        at: capture_point_now(),
                    })
                    .map_err(|error| error.to_string())?;
            } else if status != CaptureStatus::Failed {
                return Err("active microphone capture has no selected input device".to_owned());
            }
        }
        self.meter = None;
        self.touch();
        Ok(had_input || had_runtime || status != CaptureStatus::Failed)
    }

    fn refresh(&mut self) {
        if self.lifecycle.status() != CaptureStatus::Recording {
            if let Ok(permission) = CpalInput::permission_status() {
                if self.permission != permission {
                    self.permission = permission;
                    self.touch();
                }
            }
            self.refresh_input_device_directory();
        }

        let runtime_snapshot = match self.runtime.as_ref() {
            Some(runtime) => match runtime.snapshot() {
                Ok(snapshot) => Some(snapshot),
                Err(_) => {
                    if self.lifecycle.status() == CaptureStatus::Recording {
                        self.fail_active_input(
                            CaptureFailureCode::StreamStartFailed,
                            CaptureIssueCode::StreamStartFailed,
                        );
                    }
                    self.last_issue = Some(CaptureIssue {
                        code: CaptureIssueCode::StreamStartFailed,
                        device_name: self
                            .input
                            .as_ref()
                            .map(|input| input.device().name().to_owned()),
                    });
                    self.touch();
                    return;
                }
            },
            None => None,
        };
        let next_bridge = runtime_snapshot
            .as_ref()
            .map(CaptureBridgeProjection::from_snapshot);
        if self.bridge != next_bridge {
            self.bridge = next_bridge;
            self.touch();
        }

        // Before arm, the runtime is intentionally parked. Its meter and
        // ingress drops are not yet part of a committed capture timeline.
        if self.lifecycle.status() != CaptureStatus::Recording {
            return;
        }

        let Some((telemetry, device_name)) = self
            .input
            .as_ref()
            .map(|input| (input.telemetry(), input.device().name().to_owned()))
        else {
            return;
        };
        let Some(runtime_snapshot) = runtime_snapshot else {
            self.fail_active_input(
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
            );
            return;
        };
        let next_meter = CaptureMeter {
            rms_dbfs: runtime_snapshot.meter.rms_dbfs,
            peak_dbfs: runtime_snapshot.meter.peak_dbfs,
            clipping: runtime_snapshot.meter.clipping,
            dropped_packets: telemetry.dropped_packets,
        };
        if self.meter.as_ref() != Some(&next_meter) {
            self.meter = Some(next_meter);
            self.touch();
        }
        if telemetry.dropped_packets > self.observed_dropped_packets {
            self.observe_dropped_packets(telemetry.dropped_packets, capture_point_now());
            self.last_issue = Some(CaptureIssue {
                code: CaptureIssueCode::CaptureQueueOverrun,
                device_name: Some(device_name),
            });
            self.touch();
        } else {
            self.flush_queue_overrun_gap();
        }

        match telemetry.failure {
            CpalInputFailure::None => {}
            CpalInputFailure::DeviceUnavailable | CpalInputFailure::StreamInvalidated => {
                self.interrupt_active_input(CaptureIssueCode::InputDeviceUnavailable)
            }
            CpalInputFailure::PermissionDenied => self.fail_active_input(
                CaptureFailureCode::PermissionDenied,
                CaptureIssueCode::PermissionDenied,
            ),
            CpalInputFailure::Other => self.fail_active_input(
                CaptureFailureCode::StreamStartFailed,
                CaptureIssueCode::StreamStartFailed,
            ),
        }
    }

    /// The lifecycle is the source of truth only for a capture that has
    /// actually started. Before and after recording, the selectable directory
    /// owns the intended device, which lets an idle UI immediately reflect a
    /// successful user selection.
    fn selected_device_for_projection(&self) -> Option<MacOsInputDevice> {
        if self.lifecycle.status() == CaptureStatus::Recording {
            return self.lifecycle.selected_device().cloned();
        }

        let selected_uid = self.selected_device_uid.as_deref()?;
        self.devices
            .iter()
            .find(|device| device.uid() == selected_uid)
            .cloned()
    }

    /// Apply one complete successful CoreAudio directory scan. A failed scan
    /// deliberately does not call this method, preserving the last known-good
    /// directory and its current visible selection.
    fn apply_input_device_directory(&mut self, directory: InputDeviceList) {
        let next_selected_device_uid = self
            .selected_device_uid
            .as_deref()
            .filter(|selected_uid| {
                directory
                    .devices
                    .iter()
                    .any(|device| device.uid() == *selected_uid)
            })
            .map(str::to_owned)
            .or_else(|| {
                directory.default_uid.filter(|default_uid| {
                    directory
                        .devices
                        .iter()
                        .any(|device| device.uid() == default_uid)
                })
            });
        let directory_changed = self.devices != directory.devices;
        let selection_changed = self.selected_device_uid != next_selected_device_uid;

        if directory_changed {
            self.devices = directory.devices;
        }
        if selection_changed {
            self.selected_device_uid = next_selected_device_uid;
        }
        if directory_changed || selection_changed {
            self.touch();
        }
    }

    fn refresh_input_device_directory(&mut self) {
        self.apply_input_device_directory_result(CpalInput::list_input_devices());
    }

    fn apply_input_device_directory_result(&mut self, directory: Result<InputDeviceList, String>) {
        if let Ok(directory) = directory {
            self.apply_input_device_directory(directory);
        }
    }

    fn interrupt_active_input(&mut self, issue: CaptureIssueCode) {
        let Some(mut input) = self.input.take() else {
            return;
        };
        let device = input.device().clone();
        input.stop();
        self.prepared_capture = None;
        if let Some(runtime) = self.runtime.as_ref() {
            let _ = runtime.request_shutdown();
        }
        let at = capture_point_now();
        self.flush_queue_overrun_gap();
        self.record_input_device_unavailable_gap(at.clone());
        let _ = self
            .lifecycle
            .apply(MacOsCaptureEvent::InputDeviceUnavailable {
                device: device.clone(),
                at,
            });
        self.meter = None;
        self.last_issue = Some(CaptureIssue {
            code: issue,
            device_name: Some(device.name().to_owned()),
        });
        self.touch();
    }

    fn observe_dropped_packets(&mut self, dropped_packets: u64, observed_at: CapturePoint) {
        if dropped_packets <= self.observed_dropped_packets {
            self.flush_queue_overrun_gap();
            return;
        }
        self.observed_dropped_packets = dropped_packets;
        if let Some(gap) = self.active_queue_overrun_gap.as_mut() {
            if observed_at.monotonic_ns >= gap.ended_at.monotonic_ns {
                gap.ended_at = observed_at;
            }
            return;
        }
        self.active_queue_overrun_gap = Some(CaptureGap {
            started_at: observed_at.clone(),
            ended_at: observed_at,
            reason: CaptureGapReason::QueueOverrun,
        });
    }

    fn set_armed_drop_baseline(&mut self, dropped_packets: u64) {
        self.observed_dropped_packets = dropped_packets;
        // A parked runtime must not contribute a physical discontinuity to
        // the session that starts at the durable arm boundary.
        self.active_queue_overrun_gap = None;
    }

    fn record_input_device_unavailable_gap(&mut self, observed_at: CapturePoint) {
        self.record_pending_gap(CaptureGapReason::InputDeviceUnavailable, observed_at);
    }

    fn flush_queue_overrun_gap(&mut self) {
        if let Some(gap) = self.active_queue_overrun_gap.take() {
            self.enqueue_pending_gap(gap);
        }
    }

    fn record_pending_gap(&mut self, reason: CaptureGapReason, observed_at: CapturePoint) {
        self.enqueue_pending_gap(CaptureGap {
            started_at: observed_at.clone(),
            ended_at: observed_at,
            reason,
        });
    }

    fn enqueue_pending_gap(&mut self, gap: CaptureGap) {
        let can_merge_tail = self.active_gap_lease.is_none() || self.pending_gaps.len() > 1;
        if can_merge_tail {
            if let Some(previous) = self.pending_gaps.back_mut() {
                if previous.reason == gap.reason
                    && gap.started_at.monotonic_ns >= previous.ended_at.monotonic_ns
                {
                    previous.ended_at = gap.ended_at;
                    return;
                }
            }
        }

        if self.pending_gaps.len() == MAX_PENDING_GAPS {
            if self.active_gap_lease.is_some() {
                // Never evict the lease currently being persisted. This keeps
                // an SQLite retry bound to the exact gap it claimed.
                self.pending_gaps.remove(1);
            } else {
                self.pending_gaps.pop_front();
            }
        }
        self.pending_gaps.push_back(gap);
    }

    fn fail_active_input(&mut self, code: CaptureFailureCode, issue: CaptureIssueCode) {
        let device_name = self
            .input
            .as_ref()
            .map(|input| input.device().name().to_owned());
        if let Some(mut input) = self.input.take() {
            input.stop();
        }
        self.prepared_capture = None;
        if let Some(runtime) = self.runtime.as_ref() {
            let _ = runtime.request_shutdown();
        }
        self.flush_queue_overrun_gap();
        self.fail(
            capture_point_now(),
            code,
            issue,
            device_name,
            "microphone capture stream failed",
        );
    }

    fn fail_permission(&mut self, at: CapturePoint, code: CaptureFailureCode) {
        let issue = match code {
            CaptureFailureCode::PermissionRestricted => CaptureIssueCode::PermissionRestricted,
            _ => CaptureIssueCode::PermissionDenied,
        };
        self.fail(
            at,
            code,
            issue,
            None,
            "microphone permission was not granted",
        );
    }

    fn fail(
        &mut self,
        at: CapturePoint,
        code: CaptureFailureCode,
        issue: CaptureIssueCode,
        device_name: Option<String>,
        message: &str,
    ) {
        let _ = self.lifecycle.fail_with_code(at, code, message);
        self.meter = None;
        self.last_issue = Some(CaptureIssue {
            code: issue,
            device_name,
        });
        self.touch();
    }

    fn require_active_gap_lease(&self, token: u64) -> Result<&CaptureGapLease, String> {
        let lease = self
            .active_gap_lease
            .as_ref()
            .ok_or_else(|| "no capture gap delivery is active".to_owned())?;
        if lease.token != token {
            return Err("capture gap delivery token does not match the active lease".to_owned());
        }
        Ok(lease)
    }

    fn release_prepared_input(&mut self) {
        if let Some(mut input) = self.input.take() {
            input.stop();
        }
        self.prepared_capture = None;
    }

    fn retain_prearm_runtime_for_join(&mut self, runtime: NativeCaptureRuntime) {
        self.prearm_runtimes_for_join.push_back(runtime);
    }

    fn stop_input_and_request_runtime_shutdown(&mut self) -> Result<(), String> {
        self.release_prepared_input();
        if let Some(runtime) = self.runtime.as_ref() {
            runtime
                .request_shutdown()
                .map_err(|error| format!("could not stop native capture runtime: {error}"))?;
        }
        Ok(())
    }

    fn cancel_prepared_lifecycle(&mut self) -> Result<(), String> {
        if self.lifecycle.status() == CaptureStatus::AwaitingPermission {
            self.lifecycle
                .cancel_preparation(capture_point_now())
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
    }
}

fn preflight_input_sample_rate(sample_rate: u32) -> Result<(), String> {
    if SUPPORTED_CAPTURE_SAMPLE_RATES.contains(&sample_rate) {
        return Ok(());
    }
    Err(format!(
        "microphone sample rate {sample_rate} Hz is unsupported; only 16000 Hz and 48000 Hz inputs are currently supported"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};
    use std::time::{Duration as StdDuration, Instant};
    use uuid::Uuid;

    fn point(monotonic_ns: u64, milliseconds: i64) -> CapturePoint {
        CapturePoint {
            monotonic_ns,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::milliseconds(milliseconds),
        }
    }

    fn prearm_runtime() -> NativeCaptureRuntime {
        let anchor = point(1_000, 1);
        let ingress = crate::audio::CaptureIngress::new(2, 160).unwrap();
        let runtime = DispatcherRuntime::new(
            crate::audio::DispatcherRuntimeId::generate(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            anchor.clone(),
        )
        .unwrap();
        NativeCaptureRuntime::new(
            ingress,
            runtime,
            CaptureClock::new(anchor, 16_000).unwrap(),
            NativeCaptureRuntimeConfig {
                idle_wait: StdDuration::from_millis(1),
                ..NativeCaptureRuntimeConfig::default()
            },
        )
        .unwrap()
    }

    fn test_device() -> MacOsInputDevice {
        MacOsInputDevice::new("test-input", "Test microphone").unwrap()
    }

    fn input_device(uid: &str, name: &str) -> MacOsInputDevice {
        MacOsInputDevice::new(uid, name).unwrap()
    }

    fn input_device_list(
        default_uid: Option<&str>,
        devices: Vec<MacOsInputDevice>,
    ) -> InputDeviceList {
        InputDeviceList {
            devices,
            default_uid: default_uid.map(str::to_owned),
        }
    }

    fn wait_for_drained_runtime(service: &mut CaptureService) -> NativeCaptureRuntime {
        let deadline = Instant::now() + StdDuration::from_secs(1);
        loop {
            if let Some(runtime) = service.take_drained_native_runtime().unwrap() {
                return runtime;
            }
            assert!(
                Instant::now() < deadline,
                "native capture runtime did not drain after shutdown"
            );
            std::thread::sleep(StdDuration::from_millis(1));
        }
    }

    #[test]
    fn initial_projection_never_contains_pcm() {
        let mut service = CaptureService::new();
        let projection = service.projection();

        assert_eq!(projection.status, CaptureStatus::Idle);
        assert!(projection.meter.is_none());
        assert!(!serde_json::to_string(&projection)
            .unwrap()
            .contains("rmsDbfs"));
    }

    #[test]
    fn idle_projection_resolves_the_configured_device_from_the_directory() {
        let mut service = CaptureService::new();
        let built_in = input_device("built-in", "MacBook microphone");
        let usb = input_device("usb", "USB microphone");
        let built_in_uid = built_in.uid().to_owned();
        service.apply_input_device_directory(input_device_list(
            Some(&built_in_uid),
            vec![built_in, usb.clone()],
        ));
        service.selected_device_uid = Some(usb.uid().to_owned());

        assert_eq!(service.selected_device_for_projection(), Some(usb));
    }

    #[test]
    fn active_capture_keeps_the_lifecycle_device_authoritative() {
        let mut service = CaptureService::new();
        let built_in = input_device("built-in", "MacBook microphone");
        let usb = input_device("usb", "USB microphone");
        service.apply_input_device_directory(input_device_list(
            Some(built_in.uid()),
            vec![built_in.clone(), usb.clone()],
        ));
        service.selected_device_uid = Some(usb.uid().to_owned());
        service
            .lifecycle
            .apply(MacOsCaptureEvent::CaptureStarted {
                device: built_in.clone(),
                at: point(1, 1),
            })
            .unwrap();

        assert_eq!(service.selected_device_for_projection(), Some(built_in));
    }

    #[test]
    fn stale_idle_selection_falls_back_to_the_current_default_and_touches_revision() {
        let mut service = CaptureService::new();
        let built_in = input_device("built-in", "MacBook microphone");
        let usb = input_device("usb", "USB microphone");
        service.devices = vec![built_in.clone(), usb];
        service.selected_device_uid = Some("unplugged".to_owned());
        service.revision = 7;

        service.apply_input_device_directory(input_device_list(
            Some(built_in.uid()),
            vec![built_in.clone()],
        ));

        assert_eq!(service.selected_device_uid.as_deref(), Some(built_in.uid()));
        assert_eq!(service.selected_device_for_projection(), Some(built_in));
        assert_eq!(service.revision, 8);
    }

    #[test]
    fn stale_selection_change_touches_revision_even_when_the_directory_is_unchanged() {
        let mut service = CaptureService::new();
        let built_in = input_device("built-in", "MacBook microphone");
        let usb = input_device("usb", "USB microphone");
        let built_in_uid = built_in.uid().to_owned();
        let usb_uid = usb.uid().to_owned();
        service.devices = vec![built_in.clone(), usb.clone()];
        service.selected_device_uid = Some(usb_uid.clone());
        service.revision = 7;

        service.apply_input_device_directory(input_device_list(
            Some(&built_in_uid),
            vec![built_in.clone(), usb.clone()],
        ));

        assert_eq!(
            service.selected_device_uid.as_deref(),
            Some(usb_uid.as_str())
        );
        assert_eq!(service.revision, 7);

        service.selected_device_uid = Some("unplugged".to_owned());
        service.apply_input_device_directory(input_device_list(
            Some(&built_in_uid),
            vec![built_in, usb],
        ));

        assert_eq!(service.revision, 8);
    }

    #[test]
    fn failed_directory_refresh_preserves_the_last_good_devices_and_selection() {
        let mut service = CaptureService::new();
        let built_in = input_device("built-in", "MacBook microphone");
        let usb = input_device("usb", "USB microphone");
        service.devices = vec![built_in, usb.clone()];
        service.selected_device_uid = Some(usb.uid().to_owned());
        service.revision = 7;

        service.apply_input_device_directory_result(Err("CoreAudio is reconfiguring".to_owned()));

        assert_eq!(service.devices.len(), 2);
        assert_eq!(service.selected_device_for_projection(), Some(usb));
        assert_eq!(service.revision, 7);
    }

    #[test]
    fn issue_serializes_as_a_typed_code() {
        let issue = CaptureIssue {
            code: CaptureIssueCode::PermissionDenied,
            device_name: None,
        };
        assert_eq!(
            serde_json::to_string(&issue).unwrap(),
            "{\"code\":\"permission_denied\",\"deviceName\":null}"
        );
    }

    #[test]
    fn coalesces_queue_overrun_gaps_when_drops_increase() {
        let mut service = CaptureService::new();

        service.observe_dropped_packets(0, point(10, 10));
        service.observe_dropped_packets(1, point(20, 20));
        service.observe_dropped_packets(3, point(40, 40));
        assert!(service.take_pending_gaps().is_empty());

        service.flush_queue_overrun_gap();

        assert_eq!(
            service.take_pending_gaps(),
            vec![CaptureGap {
                started_at: point(20, 20),
                ended_at: point(40, 40),
                reason: CaptureGapReason::QueueOverrun,
            }]
        );
        assert!(service.take_pending_gaps().is_empty());
    }

    #[test]
    fn arm_drop_baseline_excludes_parked_queue_overruns() {
        let mut service = CaptureService::new();

        service.observe_dropped_packets(4, point(20, 20));
        service.set_armed_drop_baseline(4);
        service.observe_dropped_packets(4, point(30, 30));
        assert!(service.take_pending_gaps().is_empty());

        service.observe_dropped_packets(5, point(40, 40));
        service.flush_queue_overrun_gap();
        assert_eq!(
            service.take_pending_gaps(),
            vec![CaptureGap {
                started_at: point(40, 40),
                ended_at: point(40, 40),
                reason: CaptureGapReason::QueueOverrun,
            }]
        );
    }

    #[test]
    fn capture_gap_lease_retries_the_same_gap_until_commit() {
        let mut service = CaptureService::new();
        let gap = CaptureGap {
            started_at: point(20, 20),
            ended_at: point(40, 40),
            reason: CaptureGapReason::QueueOverrun,
        };
        service.pending_gaps.push_back(gap.clone());

        let first = service.begin_pending_gap().unwrap().unwrap();
        assert_eq!(first.gap(), &gap);
        assert!(service.take_pending_gaps().is_empty());
        assert!(service
            .commit_pending_gap(first.token().saturating_add(1))
            .is_err());

        service.abort_pending_gap(first.token()).unwrap();
        let retry = service.begin_pending_gap().unwrap().unwrap();
        assert_ne!(retry.token(), first.token());
        assert_eq!(retry.gap(), &gap);

        service.commit_pending_gap(retry.token()).unwrap();
        assert!(service.begin_pending_gap().unwrap().is_none());
    }

    #[test]
    fn leased_head_is_not_coalesced_with_a_later_gap() {
        let mut service = CaptureService::new();
        service.record_pending_gap(CaptureGapReason::QueueOverrun, point(20, 20));
        let lease = service.begin_pending_gap().unwrap().unwrap();

        service.record_pending_gap(CaptureGapReason::QueueOverrun, point(40, 40));

        assert_eq!(lease.gap().ended_at, point(20, 20));
        service.commit_pending_gap(lease.token()).unwrap();
        let later = service.begin_pending_gap().unwrap().unwrap();
        assert_eq!(later.gap().started_at, point(40, 40));
    }

    #[test]
    fn preflight_accepts_only_the_supported_native_input_rates() {
        assert!(preflight_input_sample_rate(16_000).is_ok());
        assert!(preflight_input_sample_rate(48_000).is_ok());
        assert!(preflight_input_sample_rate(44_100)
            .unwrap_err()
            .contains("44100"));
    }

    #[test]
    fn bridge_projection_serializes_only_status_and_compact_metrics() {
        let bridge = CaptureBridgeProjection::from_snapshot(&NativeCaptureRuntimeSnapshot {
            status: NativeCaptureRuntimeStatus::Armed,
            dispatcher_status: crate::audio::DispatcherStatus::Running,
            armed: true,
            shutdown_requested: false,
            worker_finished: false,
            meter: crate::audio::DispatcherMeter::default(),
            metrics: AsrQueueMetrics {
                ingress_packets_consumed: 4,
                job_queue_depth: 1,
                unavailable_engine_outcomes: 2,
                ..AsrQueueMetrics::default()
            },
        });

        let encoded = serde_json::to_string(&bridge).unwrap();
        assert!(encoded.contains("unavailableEngineOutcomes"));
        assert!(!encoded.contains("samples"));
        assert!(!encoded.contains("text"));
    }

    #[test]
    fn prearm_failure_runtime_is_explicitly_transferred_for_lock_external_join() {
        let mut service = CaptureService::new();
        let runtime = prearm_runtime();
        runtime.abort_before_arm().unwrap();
        service.retain_prearm_runtime_for_join(runtime);

        let mut cleanup = service
            .take_prearm_runtime_for_join()
            .expect("aborted pre-arm runtime stays owned until the caller retrieves it");
        cleanup.join_after_abort().unwrap();

        assert!(service.take_prearm_runtime_for_join().is_none());
    }

    #[test]
    fn stop_cleans_up_a_parked_runtime_after_input_was_already_released() {
        let mut service = CaptureService::new();
        service
            .lifecycle
            .begin_permission_resolution(point(10, 10))
            .unwrap();
        service.runtime = Some(prearm_runtime());

        assert!(service.stop().unwrap());
        assert_eq!(service.lifecycle.status(), CaptureStatus::Idle);

        let mut runtime = service
            .take_prearm_runtime_for_join()
            .expect("the parked runtime stays available for lock-external cleanup");
        runtime.join_after_abort().unwrap();
        assert!(!service.stop().unwrap());
    }

    #[test]
    fn stop_retries_runtime_shutdown_after_input_was_already_released() {
        let mut service = CaptureService::new();
        let device = test_device();
        service
            .lifecycle
            .apply(MacOsCaptureEvent::CaptureStarted {
                device,
                at: point(10, 10),
            })
            .unwrap();
        let runtime = prearm_runtime();
        runtime.arm().unwrap();
        service.runtime = Some(runtime);

        assert!(service.stop().unwrap());
        assert_eq!(service.lifecycle.status(), CaptureStatus::Idle);
        // The first stop may have already stopped CPAL. Keep signalling the
        // native worker until ownership is handed off for its final join.
        assert!(service.stop().unwrap());

        let mut runtime = wait_for_drained_runtime(&mut service);
        assert!(runtime.join_if_drained().unwrap());
        assert!(!service.stop().unwrap());
    }

    #[test]
    fn reports_input_device_unavailable_without_returning_pcm() {
        let mut service = CaptureService::new();

        service.record_input_device_unavailable_gap(point(70, 70));
        let gaps = service.take_pending_gaps();

        assert_eq!(
            gaps,
            vec![CaptureGap {
                started_at: point(70, 70),
                ended_at: point(70, 70),
                reason: CaptureGapReason::InputDeviceUnavailable,
            }]
        );
        assert!(!serde_json::to_string(&gaps).unwrap().contains("samples"));
    }

    #[test]
    fn keeps_pending_gaps_bounded_and_preserves_recent_events() {
        let mut service = CaptureService::new();

        for index in 0..(MAX_PENDING_GAPS + 3) {
            let reason = if index % 2 == 0 {
                CaptureGapReason::QueueOverrun
            } else {
                CaptureGapReason::SystemSleep
            };
            service.record_pending_gap(reason, point(index as u64, index as i64));
        }

        let gaps = service.take_pending_gaps();
        assert_eq!(gaps.len(), MAX_PENDING_GAPS);
        assert_eq!(gaps.first().map(|gap| gap.started_at.monotonic_ns), Some(3));
        assert_eq!(gaps.last().map(|gap| gap.ended_at.monotonic_ns), Some(18));
    }
}
