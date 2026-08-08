//! Application-owned microphone capture lifecycle.
//!
//! `CaptureService` owns the native stream and only exposes compact metadata.
//! Raw PCM stays inside the pre-allocated ingress and its meter worker.

use super::{
    capture_point_now, CaptureFailureCode, CaptureGap, CaptureGapReason, CaptureLifecycle,
    CapturePoint, CaptureStatus, CpalInput, CpalInputFailure, MacOsCaptureEvent, MacOsInputDevice,
    MicrophonePermission,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_PENDING_GAPS: usize = 16;

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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureProjection {
    pub revision: u64,
    pub status: CaptureStatus,
    pub permission: MicrophonePermission,
    pub selected_device: Option<MacOsInputDevice>,
    pub devices: Vec<MacOsInputDevice>,
    pub meter: Option<CaptureMeter>,
    pub last_issue: Option<CaptureIssue>,
}

#[derive(Clone, Debug)]
pub struct CaptureStart {
    pub anchor: CapturePoint,
    pub device: MacOsInputDevice,
    pub sample_rate: u32,
    pub channels: u16,
}

pub struct CaptureService {
    lifecycle: CaptureLifecycle,
    permission: MicrophonePermission,
    devices: Vec<MacOsInputDevice>,
    selected_device_uid: Option<String>,
    meter: Option<CaptureMeter>,
    last_issue: Option<CaptureIssue>,
    revision: u64,
    observed_dropped_packets: u64,
    pending_gaps: VecDeque<CaptureGap>,
    active_queue_overrun_gap: Option<CaptureGap>,
    input: Option<CpalInput>,
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
            last_issue: None,
            revision: 0,
            observed_dropped_packets: 0,
            pending_gaps: VecDeque::with_capacity(MAX_PENDING_GAPS),
            active_queue_overrun_gap: None,
            input: None,
        }
    }

    pub fn projection(&mut self) -> CaptureProjection {
        self.refresh();
        CaptureProjection {
            revision: self.revision,
            status: self.lifecycle.status(),
            permission: self.permission,
            selected_device: self.lifecycle.selected_device().cloned(),
            devices: self.devices.clone(),
            meter: self.meter.clone(),
            last_issue: self.last_issue.clone(),
        }
    }

    /// Return and clear bounded capture discontinuities for durable storage.
    ///
    /// Gap metadata is deliberately compact and never contains PCM samples.
    pub fn take_pending_gaps(&mut self) -> Vec<CaptureGap> {
        self.pending_gaps.drain(..).collect()
    }

    pub fn select_input_device(&mut self, device_uid: String) -> Result<CaptureProjection, String> {
        self.refresh();
        if self.lifecycle.status() == CaptureStatus::Recording {
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

    pub fn start(&mut self) -> Result<CaptureStart, String> {
        self.refresh();
        if self.lifecycle.status() == CaptureStatus::Recording {
            return Err("microphone capture is already active".to_owned());
        }

        let requested_at = capture_point_now();
        self.lifecycle
            .begin_permission_resolution(requested_at.clone())
            .map_err(|error| error.to_string())?;
        self.last_issue = None;
        self.meter = None;
        self.observed_dropped_packets = 0;
        // Gaps belong to the stream that produced them and must never leak
        // into a later session after an interrupted stream is restarted.
        self.pending_gaps.clear();
        self.active_queue_overrun_gap = None;
        self.touch();

        self.permission = CpalInput::permission_status().map_err(|error| {
            let _ = self.lifecycle.fail_with_code(
                requested_at.clone(),
                CaptureFailureCode::External,
                "could not inspect microphone permission",
            );
            self.last_issue = Some(CaptureIssue {
                code: CaptureIssueCode::StreamStartFailed,
                device_name: None,
            });
            self.touch();
            error
        })?;
        if self.permission == MicrophonePermission::Denied {
            self.fail_permission(requested_at, CaptureFailureCode::PermissionDenied);
            return Err("microphone permission is denied".to_owned());
        }
        if self.permission == MicrophonePermission::Restricted {
            self.fail_permission(requested_at, CaptureFailureCode::PermissionRestricted);
            return Err("microphone permission is restricted by macOS".to_owned());
        }

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
        let (input, anchor) = match CpalInput::start(selected_device_uid) {
            Ok(started) => started,
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
        self.permission = MicrophonePermission::Granted;
        self.lifecycle
            .apply(MacOsCaptureEvent::CaptureStarted {
                device: device.clone(),
                at: anchor.clone(),
            })
            .map_err(|error| error.to_string())?;
        self.input = Some(input);
        self.touch();

        Ok(CaptureStart {
            anchor,
            device,
            sample_rate,
            channels,
        })
    }

    pub fn stop(&mut self) -> Result<bool, String> {
        self.refresh();
        self.flush_queue_overrun_gap();
        let Some(mut input) = self.input.take() else {
            return Ok(false);
        };
        let device = input.device().clone();
        input.stop();
        if matches!(
            self.lifecycle.status(),
            CaptureStatus::Recording | CaptureStatus::Interrupted
        ) {
            self.lifecycle
                .apply(MacOsCaptureEvent::CaptureStopped {
                    device,
                    at: capture_point_now(),
                })
                .map_err(|error| error.to_string())?;
        }
        self.meter = None;
        self.touch();
        Ok(true)
    }

    fn refresh(&mut self) {
        if self.lifecycle.status() != CaptureStatus::Recording {
            if let Ok(permission) = CpalInput::permission_status() {
                if self.permission != permission {
                    self.permission = permission;
                    self.touch();
                }
            }
            if let Ok(devices) = CpalInput::list_input_devices() {
                if self.devices != devices.devices {
                    self.devices = devices.devices;
                    self.touch();
                }
                if self.selected_device_uid.is_none() {
                    self.selected_device_uid = devices.default_uid;
                }
            }
        }

        let Some((telemetry, device_name)) = self
            .input
            .as_ref()
            .map(|input| (input.telemetry(), input.device().name().to_owned()))
        else {
            return;
        };
        let next_meter = CaptureMeter {
            rms_dbfs: telemetry.rms_dbfs,
            peak_dbfs: telemetry.peak_dbfs,
            clipping: telemetry.clipping,
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

    fn interrupt_active_input(&mut self, issue: CaptureIssueCode) {
        let Some(mut input) = self.input.take() else {
            return;
        };
        let device = input.device().clone();
        input.stop();
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
        if let Some(previous) = self.pending_gaps.back_mut() {
            if previous.reason == gap.reason
                && gap.started_at.monotonic_ns >= previous.ended_at.monotonic_ns
            {
                previous.ended_at = gap.ended_at;
                return;
            }
        }

        if self.pending_gaps.len() == MAX_PENDING_GAPS {
            self.pending_gaps.pop_front();
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

    fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};

    fn point(monotonic_ns: u64, milliseconds: i64) -> CapturePoint {
        CapturePoint {
            monotonic_ns,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::milliseconds(milliseconds),
        }
    }

    #[test]
    fn initial_projection_never_contains_pcm() {
        let mut service = CaptureService::new();
        let projection = service.projection();

        assert_eq!(projection.status, CaptureStatus::Idle);
        assert!(projection.meter.is_none());
        assert!(
            serde_json::to_string(&projection)
                .unwrap()
                .contains("rmsDbfs")
                == false
        );
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
