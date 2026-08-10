#[cfg(all(target_os = "macos", not(test)))]
use crate::audio::{capture_point_now, CaptureStart, DispatcherRuntimeId, NativeInferenceEngines};
use crate::audio::{AsrJobMetadata, AsrOutcome, CapturePoint, DispatcherRuntime};
#[cfg(target_os = "macos")]
use crate::audio::{CaptureGap, CaptureProjection, CaptureService};
#[cfg(any(test, debug_assertions))]
use crate::audio::{DevelopmentMockProgress, DevelopmentMockRunner};
use crate::audit::{
    AsrFinalAuditPayload, AsrFinalIdempotencyBinding, AuditKind, AuditStore, AuditStoreError,
    AuditTrail,
};
#[cfg(target_os = "macos")]
use crate::domain::CaptureSegment;
use crate::domain::{
    CaptureSession, DataCategory, SpeakerCluster, SpeakerClusterCreatedAuditPayload,
    SpeakerClusterLabelRevision, SpeakerClusterRecord, TranscriptModelProvenance,
    TranscriptRevision, TranscriptSource, TranscriptSpan, TranscriptTiming,
};
use crate::inference::asr::MAX_ASR_EMISSIONS_PER_REQUEST;
use crate::inference::model_registry::{
    LocalModelKind, ModelImportRequest, ModelRegistry, RegisteredModel,
};
use crate::inference::{
    is_whisper_cpp_compatible_input_format, AsrResponse, InferenceGap, InferenceGapReason,
    InferenceGapStage, MappedTranscriptEmission, TranscriptEmission, TranscriptEmissionKind,
    TranscriptEmissionMapper,
};
#[cfg(all(target_os = "macos", not(test)))]
use crate::inference::{WebRtcVad, WhisperCppAsrEngine};
use crate::policy::{EgressApproval, EgressPolicy, EgressRequest, PolicyDecision, PolicyReason};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrivacyStatus {
    pub local_only: bool,
    pub egress_enabled: bool,
    pub active_egress_approvals: usize,
    pub recording_session_id: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Ready,
    Blocked,
    Completed,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAction {
    pub id: Uuid,
    pub title: String,
    pub detail: String,
    pub status: ActionStatus,
    pub kind: String,
}

/// A compact reference to a final transcript projection changed by a manual
/// speaker-catalog operation. The WebView reloads the full timeline only when
/// this list is non-empty.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerSpanRef {
    pub id: Uuid,
    pub revision: u32,
}

/// Durable result returned by every manual speaker-catalog command.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerOperationResult {
    pub clusters: Vec<SpeakerCluster>,
    pub updated_spans: Vec<SpeakerSpanRef>,
}

/// A compact notification that a native final transcript was durably added to
/// a session timeline. The WebView must reload the timeline to obtain the
/// transcript itself; this notification intentionally contains no content or
/// capture payload.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalTranscriptProjection {
    pub session_id: Uuid,
    pub revision: u64,
}

/// The explicitly chosen local speech-recognition model for this app run.
///
/// This is intentionally not persisted: reopening WordCovenant requires a
/// fresh visible choice before microphone recording can load a local model.
/// It contains an opaque model ID only, never a file path, audio, or model
/// artifact bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveLocalAsrProfile {
    pub model_id: Uuid,
}

#[derive(Default)]
struct FinalTranscriptProjectionState {
    revision: u64,
    latest: Option<FinalTranscriptProjection>,
}

/// The application-owned lifecycle fence for one native dispatcher generation.
///
/// A capture service can only expose one live native runtime, but this fence
/// deliberately lives beside session state. It prevents a delayed worker
/// outcome from an earlier generation from entering a restarted session.
#[derive(Clone, Debug)]
struct NativeRuntimeFence {
    runtime: DispatcherRuntime,
    phase: NativeRuntimePhase,
    capture_input_stopped_audited: bool,
    capture_stop_point: Option<CapturePoint>,
}

#[derive(Clone, Debug)]
struct NativeCaptureStop {
    runtime: DispatcherRuntime,
    point: CapturePoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeRuntimePhase {
    Active,
    Closing,
    Handoff,
    Drained,
}

pub struct AppState {
    started_at: Instant,
    sessions: Mutex<BTreeMap<Uuid, CaptureSession>>,
    timelines: Mutex<BTreeMap<Uuid, Vec<TranscriptSpan>>>,
    actions: Mutex<Vec<AgentAction>>,
    policy: Mutex<EgressPolicy>,
    audit_trail: Mutex<AuditTrail>,
    audit_store: Mutex<AuditStore>,
    model_registry: Mutex<ModelRegistry>,
    active_local_asr_profile: Mutex<Option<ActiveLocalAsrProfile>>,
    inference_mapper: Mutex<TranscriptEmissionMapper>,
    final_transcript_projection: Mutex<FinalTranscriptProjectionState>,
    native_runtime: Mutex<Option<NativeRuntimeFence>>,
    model_root: Option<PathBuf>,
    #[cfg(target_os = "macos")]
    capture_service: Mutex<CaptureService>,
    /// Serializes the externally visible start/stop transition. CPAL must not
    /// be stopped between the durable start bundle and dispatcher arm.
    #[cfg(target_os = "macos")]
    native_capture_lifecycle: Mutex<()>,
    #[cfg(any(test, debug_assertions))]
    development_mock: Mutex<Option<DevelopmentMockRunner>>,
}

impl AppState {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AuditStoreError> {
        let database_path = path.as_ref();
        let model_root = database_path
            .parent()
            .filter(|parent| parent.is_absolute())
            .map(|parent| parent.join("models"));
        let audit_store = AuditStore::open_path(database_path)?;
        if !audit_store.verify()? {
            return Err(AuditStoreError::Integrity);
        }
        let audit_trail = AuditTrail::from_events(audit_store.list()?);
        if !audit_trail.verify() {
            return Err(AuditStoreError::Integrity);
        }

        Self::from_audit_store(audit_store, audit_trail, model_root)
    }

    #[cfg(test)]
    pub fn in_memory() -> Self {
        let audit_store = AuditStore::open_in_memory().expect("in-memory audit store opens");
        Self::from_audit_store(audit_store, AuditTrail::default(), None)
            .expect("in-memory transcript projection loads")
    }

    fn from_audit_store(
        audit_store: AuditStore,
        audit_trail: AuditTrail,
        model_root: Option<PathBuf>,
    ) -> Result<Self, AuditStoreError> {
        let timelines = timeline_projections(audit_store.list_all_transcript_revisions()?);
        let persisted_models = audit_store.list_local_models()?;
        let model_registry = if let Some(model_root) = model_root.as_deref() {
            ModelRegistry::from_persisted(model_root, persisted_models)
        } else if persisted_models.is_empty() {
            Ok(ModelRegistry::new())
        } else {
            Err(crate::inference::model_registry::ModelRegistryError::ManagedRootNotConfigured)
        }
        .map_err(|error| AuditStoreError::InvalidModelMetadata {
            field: "persisted registry",
            value: error.to_string(),
        })?;
        Ok(Self {
            started_at: Instant::now(),
            sessions: Mutex::new(BTreeMap::new()),
            timelines: Mutex::new(timelines),
            actions: Mutex::new(Vec::new()),
            policy: Mutex::new(EgressPolicy::default()),
            audit_trail: Mutex::new(audit_trail),
            audit_store: Mutex::new(audit_store),
            model_registry: Mutex::new(model_registry),
            active_local_asr_profile: Mutex::new(None),
            inference_mapper: Mutex::new(TranscriptEmissionMapper::default()),
            final_transcript_projection: Mutex::new(FinalTranscriptProjectionState::default()),
            native_runtime: Mutex::new(None),
            model_root,
            #[cfg(target_os = "macos")]
            capture_service: Mutex::new(CaptureService::new()),
            #[cfg(target_os = "macos")]
            native_capture_lifecycle: Mutex::new(()),
            #[cfg(any(test, debug_assertions))]
            development_mock: Mutex::new(None),
        })
    }

    pub fn privacy_status(&self) -> Result<PrivacyStatus, String> {
        let recording_session_id = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?
            .values()
            .find(|session| matches!(session.state, crate::domain::SessionState::Recording))
            .map(|session| session.id);
        let (egress_enabled, active_egress_approvals) = {
            let policy = self
                .policy
                .lock()
                .map_err(|_| "policy state lock poisoned".to_owned())?;
            let now = Utc::now();
            let active_egress_approvals = policy
                .approvals()
                .iter()
                .filter(|approval| {
                    approval.revoked_at.is_none()
                        && approval
                            .expires_at
                            .is_none_or(|expires_at| expires_at > now)
                })
                .count();
            (policy.egress_enabled(), active_egress_approvals)
        };

        Ok(PrivacyStatus {
            local_only: !egress_enabled || active_egress_approvals == 0,
            egress_enabled,
            active_egress_approvals,
            recording_session_id,
        })
    }

    pub fn set_egress_enabled(&self, enabled: bool) -> Result<PrivacyStatus, String> {
        let mut policy = self
            .policy
            .lock()
            .map_err(|_| "policy state lock poisoned".to_owned())?;
        if policy.egress_enabled() != enabled {
            self.record_audit(
                AuditKind::EgressSettingChanged,
                self.monotonic_ns(),
                &serde_json::json!({ "enabled": enabled }),
            )?;
            policy.set_egress_enabled(enabled);
        }
        drop(policy);

        self.privacy_status()
    }

    pub fn start_session(&self) -> Result<CaptureSession, String> {
        #[cfg(test)]
        {
            return self.start_session_with_active(false);
        }

        #[cfg(not(test))]
        self.start_microphone_session()
    }

    #[cfg(target_os = "macos")]
    pub fn capture_projection(&self) -> Result<CaptureProjection, String> {
        // A failed SessionStopped transaction leaves the dispatcher fence in
        // Drained state so the exact capture-clock stop point is retryable.
        // Projection polling is the normal native heartbeat, so use it to
        // finish that narrow, durable-only tail before returning stale live
        // state to the frontend.
        self.retry_drained_native_session_stop_from_projection()?;
        let projection = self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .projection();
        if let Some(session) = self.active_recording_session()? {
            self.pump_capture_gaps(session.id)?;
            self.pump_native_outcomes()?;
        }
        if matches!(
            projection.status,
            crate::audio::CaptureStatus::Interrupted | crate::audio::CaptureStatus::Failed
        ) && self.active_recording_session()?.is_some()
        {
            let _ = self.stop_session()?;
        }
        Ok(projection)
    }

    #[cfg(target_os = "macos")]
    pub fn select_input_device(&self, device_uid: String) -> Result<CaptureProjection, String> {
        self.capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .select_input_device(device_uid)
    }

    #[cfg(target_os = "macos")]
    #[cfg(not(test))]
    fn start_microphone_session(&self) -> Result<CaptureSession, String> {
        let native_lifecycle = self
            .native_capture_lifecycle
            .lock()
            .map_err(|_| "native capture lifecycle lock poisoned".to_owned())?;
        if let Some(active) = self.active_live_session()? {
            if let Some(fence) = self.native_runtime_fence()? {
                if fence.runtime.session_id == active.id
                    && fence.phase != NativeRuntimePhase::Active
                {
                    return Err("a native capture session is still starting or draining".to_owned());
                }
            }
            if matches!(active.state, crate::domain::SessionState::Recording) {
                return Ok(active);
            }
            return Err("a native capture session is still starting or draining".to_owned());
        }

        // Resolve and load the user-selected local model before touching the
        // microphone. A missing, tampered, incompatible, or unreadable model
        // therefore fails closed without starting a capture session.
        let inference_engines = self.build_active_local_inference_engines()?;

        let preparation = self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .prepare()?;
        let session_id = Uuid::new_v4();
        let runtime = DispatcherRuntime::new(
            DispatcherRuntimeId::generate(),
            session_id,
            Uuid::new_v4(),
            preparation.anchor.clone(),
        )?;
        let capture_start = match self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .activate_with_runtime_and_engines(runtime.clone(), inference_engines)
        {
            Ok(capture_start) => capture_start,
            Err(error) => {
                return Err(with_staged_capture_cleanup(
                    error,
                    self.abort_staged_native_capture(),
                ));
            }
        };
        if capture_start.runtime != runtime {
            let cleanup = self.abort_staged_native_capture();
            return Err(with_staged_capture_cleanup(
                "activated native runtime does not match its requested generation".to_owned(),
                cleanup,
            ));
        }

        let _staged_session = match self.commit_native_capture_start(session_id, &capture_start) {
            Ok(session) => session,
            Err(error) => {
                return Err(with_staged_capture_cleanup(
                    error,
                    self.abort_staged_native_capture(),
                ));
            }
        };
        let arm_result = {
            let mut service = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?;
            service.arm_after_commit()
        };
        let publish_result = arm_result.and_then(|armed_start| {
            if armed_start.runtime != capture_start.runtime {
                return Err(
                    "armed native runtime does not match its committed generation".to_owned(),
                );
            }
            self.publish_native_capture_recording(&capture_start.runtime)
        });
        // `stop_session` takes the same transition lock. Release it before
        // entering the standard post-commit drain path for an arm failure.
        drop(native_lifecycle);
        match publish_result {
            Ok(session) => Ok(session),
            Err(error) => Err(with_staged_capture_cleanup(
                error,
                self.stop_session().map(|_| ()),
            )),
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[cfg(not(test))]
    fn start_microphone_session(&self) -> Result<CaptureSession, String> {
        Err("microphone capture is only available on macOS".to_owned())
    }

    fn active_recording_session(&self) -> Result<Option<CaptureSession>, String> {
        self.sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())
            .map(|sessions| {
                sessions
                    .values()
                    .find(|session| matches!(session.state, crate::domain::SessionState::Recording))
                    .cloned()
            })
    }

    fn active_live_session(&self) -> Result<Option<CaptureSession>, String> {
        self.sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())
            .map(|sessions| {
                sessions
                    .values()
                    .find(|session| {
                        matches!(
                            session.state,
                            crate::domain::SessionState::Starting
                                | crate::domain::SessionState::Recording
                        )
                    })
                    .cloned()
            })
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn commit_native_capture_start(
        &self,
        session_id: Uuid,
        capture_start: &CaptureStart,
    ) -> Result<CaptureSession, String> {
        if capture_start.runtime.session_id != session_id {
            return Err(
                "native capture start session does not match its dispatcher runtime".to_owned(),
            );
        }
        if capture_start.runtime.capture_anchor != capture_start.anchor {
            return Err(
                "native capture start anchor does not match its dispatcher runtime".to_owned(),
            );
        }
        validate_dispatcher_runtime(&capture_start.runtime)?;
        let session = CaptureSession::begin_starting_with_id(
            session_id,
            capture_start.anchor.monotonic_ns,
            capture_start.anchor.wall_clock,
        )?;
        let segment = CaptureSegment::new_with_id(
            capture_start.runtime.capture_segment_id,
            session.id,
            capture_start.device.uid(),
            capture_start.device.name(),
            capture_start.sample_rate,
            capture_start.channels,
            capture_start.anchor.monotonic_ns,
            capture_start.anchor.wall_clock,
        )?;
        let input_started_payload = serde_json::json!({
            "sessionId": session.id,
            "runtimeId": capture_start.runtime.id.as_uuid(),
            "captureSegmentId": segment.id,
            "deviceUid": capture_start.device.uid(),
            "deviceName": capture_start.device.name(),
            "sampleRate": capture_start.sample_rate,
            "channels": capture_start.channels,
            "anchor": capture_start.anchor,
        });

        // Acquire every fallible projection lock before committing the audit
        // bundle. A failed staged start then leaves neither an active in-memory
        // session nor any partial durable start records. The native runtime
        // fence is acquired here too, so registering its generation cannot
        // fail after the immutable start bundle has committed.
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        if native_runtime.is_some() {
            return Err("a native dispatcher runtime is already registered".to_owned());
        }
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?;
        if sessions
            .values()
            .any(|session| matches!(session.state, crate::domain::SessionState::Recording))
        {
            return Err(
                "a recording session became active while microphone capture was preparing"
                    .to_owned(),
            );
        }
        let mut timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let mut staged_trail = trail.clone();
        let session_started = staged_trail
            .next_event(
                Some(session.id),
                None,
                AuditKind::SessionStarted,
                session.started_monotonic_ns,
                session.started_at,
                &session,
            )
            .map_err(|error| format!("could not serialize capture session start: {error}"))?;
        assert!(staged_trail.append_event(session_started.clone()));
        let segment_recorded = staged_trail
            .next_event(
                Some(session.id),
                None,
                AuditKind::CaptureSegmentRecorded,
                segment.anchor_monotonic_ns,
                segment.anchor_wall_clock,
                &segment,
            )
            .map_err(|error| format!("could not serialize capture segment: {error}"))?;
        assert!(staged_trail.append_event(segment_recorded.clone()));
        let input_started = staged_trail
            .next_event(
                Some(session.id),
                None,
                AuditKind::CaptureInputStarted,
                capture_start.anchor.monotonic_ns,
                capture_start.anchor.wall_clock,
                &input_started_payload,
            )
            .map_err(|error| format!("could not serialize capture input start: {error}"))?;

        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append_capture_start_bundle_with_audit(
                &session,
                &segment,
                &session_started,
                &segment_recorded,
                &input_started,
                &input_started_payload,
            )
            .map_err(|error| format!("could not persist native capture start: {error}"))?;
        assert!(trail.append_event(session_started));
        assert!(trail.append_event(segment_recorded));
        assert!(trail.append_event(input_started));
        timelines.entry(session.id).or_default();
        sessions.insert(session.id, session.clone());
        *native_runtime = Some(NativeRuntimeFence {
            runtime: capture_start.runtime.clone(),
            phase: NativeRuntimePhase::Active,
            capture_input_stopped_audited: false,
            capture_stop_point: None,
        });
        Ok(session)
    }

    #[cfg(all(target_os = "macos", test))]
    fn record_capture_segment(&self, segment: &CaptureSegment) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(
                Some(segment.session_id),
                None,
                AuditKind::CaptureSegmentRecorded,
                segment.anchor_monotonic_ns,
                segment.anchor_wall_clock,
                segment,
            )
            .map_err(|error| format!("could not serialize capture segment: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append_capture_segment_with_audit(&event, segment)
            .map_err(|error| format!("could not persist capture segment: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified capture segment event".to_owned());
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn record_capture_gap(&self, session_id: Uuid, gap: &CaptureGap) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(
                Some(session_id),
                None,
                AuditKind::CaptureGapRecorded,
                gap.ended_at.monotonic_ns,
                gap.ended_at.wall_clock,
                gap,
            )
            .map_err(|error| format!("could not serialize capture gap: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append_capture_gap_with_audit(&event, session_id, gap)
            .map_err(|error| format!("could not persist capture gap: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified capture gap event".to_owned());
        }
        Ok(())
    }

    /// Persist every currently claimable physical discontinuity. Claim and
    /// acknowledgement hold the service mutex briefly; SQLite and audit work
    /// deliberately occur after that mutex is released.
    #[cfg(target_os = "macos")]
    fn pump_capture_gaps(&self, session_id: Uuid) -> Result<usize, String> {
        let mut persisted = 0;
        loop {
            let lease = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?
                .begin_pending_gap()?;
            let Some(lease) = lease else {
                return Ok(persisted);
            };
            let token = lease.token();
            let persistence = self.record_capture_gap(session_id, lease.gap());
            match persistence {
                Ok(()) => {
                    self.capture_service
                        .lock()
                        .map_err(|_| "capture service lock poisoned".to_owned())?
                        .commit_pending_gap(token)?;
                    persisted += 1;
                }
                Err(error) => {
                    let abort = self
                        .capture_service
                        .lock()
                        .map_err(|_| "capture service lock poisoned".to_owned())?
                        .abort_pending_gap(token);
                    return match abort {
                        Ok(()) => Err(error),
                        Err(abort_error) => Err(format!(
                            "{error}; could not release capture gap delivery for retry: {abort_error}"
                        )),
                    };
                }
            }
        }
    }

    /// Persist every currently claimable native outcome under the currently
    /// registered fence. Native dispatcher locks are never held while a
    /// transcript, gap, or audit event is written to SQLite.
    #[cfg(target_os = "macos")]
    fn pump_native_outcomes(&self) -> Result<usize, String> {
        let Some(runtime) = self.native_runtime_context()? else {
            return Ok(0);
        };
        let service_runtime = self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .runtime_context()?;
        if service_runtime.as_ref() != Some(&runtime) {
            return Err("native capture service runtime does not match the state fence".to_owned());
        }

        let mut persisted = 0;
        loop {
            let lease = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?
                .begin_native_outcome()?;
            let Some(lease) = lease else {
                return Ok(persisted);
            };
            let token = lease.token();
            let persistence = self.persist_native_outcome(&runtime, lease.outcome());
            match persistence {
                Ok(_) => {
                    self.capture_service
                        .lock()
                        .map_err(|_| "capture service lock poisoned".to_owned())?
                        .commit_native_outcome(token)?;
                    persisted += 1;
                }
                Err(error) => {
                    let abort = self
                        .capture_service
                        .lock()
                        .map_err(|_| "capture service lock poisoned".to_owned())?
                        .abort_native_outcome(token);
                    return match abort {
                        Ok(()) => Err(error),
                        Err(abort_error) => Err(format!(
                            "{error}; could not release native outcome delivery for retry: {abort_error}"
                        )),
                    };
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn abort_staged_native_capture(&self) -> Result<(), String> {
        let aborted_runtime = self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .abort_after_failed_commit();
        let mut failures = Vec::new();
        match aborted_runtime {
            Ok(Some(mut runtime)) => {
                if let Err(error) = runtime.join_after_abort() {
                    failures.push(format!(
                        "could not join aborted native capture runtime: {error}"
                    ));
                }
            }
            Ok(None) => {}
            Err(error) => failures.push(error),
        }

        // Activation can fail after constructing its parked worker but before
        // returning a CaptureStart. CaptureService retains that worker until
        // it is explicitly moved here and joined outside its mutex.
        loop {
            let runtime = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?
                .take_prearm_runtime_for_join();
            let Some(mut runtime) = runtime else {
                break;
            };
            if let Err(error) = runtime.join_after_abort() {
                failures.push(format!(
                    "could not join aborted native capture runtime: {error}"
                ));
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn stop_native_capture_before_session_stop(&self) -> Result<Option<NativeCaptureStop>, String> {
        let Some(fence) = self.native_runtime_fence()? else {
            let service_runtime = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?
                .runtime_context()?;
            if service_runtime.is_some() {
                return Err("native capture runtime is missing its state fence".to_owned());
            }
            return Ok(None);
        };
        let runtime = fence.runtime;
        match fence.phase {
            NativeRuntimePhase::Drained => {
                return self
                    .finish_drained_native_runtime(&runtime)
                    .map(|point| Some(NativeCaptureStop { runtime, point }))
            }
            NativeRuntimePhase::Handoff => {
                return Err("native capture shutdown handoff is already in progress".to_owned())
            }
            NativeRuntimePhase::Active | NativeRuntimePhase::Closing => {}
        }

        self.begin_native_runtime_shutdown(&runtime)?;
        self.capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .stop()?;
        self.capture_native_stop_point(&runtime, capture_point_now().monotonic_ns)?;

        loop {
            self.pump_capture_gaps(runtime.session_id)?;
            self.pump_native_outcomes()?;

            // Claim state-side ownership before removing the runtime from the
            // service. Projection polling then sees an explicit handoff phase
            // rather than treating the short transfer as a stale generation.
            self.begin_native_runtime_handoff(&runtime)?;
            let drained_runtime = match self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?
                .take_drained_native_runtime()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    self.restore_native_runtime_closing(&runtime)?;
                    return Err(error);
                }
            };
            let Some(mut drained_runtime) = drained_runtime else {
                self.restore_native_runtime_closing(&runtime)?;
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            };

            // `take_drained_native_runtime` verifies the dispatcher is
            // terminal before moving it. Record that ownership change first;
            // a worker panic during join is therefore retryable without a
            // service/fence mismatch.
            self.mark_native_runtime_drained(&runtime)?;
            let joined = drained_runtime
                .join_if_drained()
                .map_err(|error| format!("could not join native capture runtime: {error}"))?;
            if !joined {
                return Err("native capture runtime was removed before it drained".to_owned());
            }
            return self
                .finish_drained_native_runtime(&runtime)
                .map(|point| Some(NativeCaptureStop { runtime, point }));
        }
    }

    fn finish_drained_native_runtime(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<CapturePoint, String> {
        // All queue-owned results are terminal before we discard transient
        // mapper state. Keeping the Drained fence through the two following
        // durable steps makes a failed stop safe to retry.
        let capture_stop_point = self.native_capture_stop_point(runtime)?;
        self.clear_local_asr_session(runtime.session_id)?;
        if self.native_runtime_needs_capture_input_stop_event(runtime)? {
            self.record_native_capture_input_stopped(runtime, &capture_stop_point)?;
            self.mark_native_capture_input_stopped(runtime)?;
        }
        Ok(capture_stop_point)
    }

    /// Complete the durable tail of a native stop that had already drained
    /// before an earlier `SessionStopped` write failed. This is deliberately
    /// driven from the compact projection poll: users otherwise have no
    /// visible action that can release a live session whose microphone is
    /// already idle.
    #[cfg(target_os = "macos")]
    fn retry_drained_native_session_stop_from_projection(&self) -> Result<bool, String> {
        let _native_lifecycle = self
            .native_capture_lifecycle
            .lock()
            .map_err(|_| "native capture lifecycle lock poisoned".to_owned())?;
        let fence = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?
            .clone();
        let Some(fence) = fence else {
            return Ok(false);
        };
        if fence.phase != NativeRuntimePhase::Drained {
            return Ok(false);
        }

        let Some(session) = self.active_live_session()? else {
            return Ok(false);
        };
        if session.id != fence.runtime.session_id {
            return Err("drained native dispatcher does not match the live session".to_owned());
        }

        let stop_point = self.finish_drained_native_runtime(&fence.runtime)?;
        let stopped = self.finish_capture_session_at(stop_point)?;
        if stopped.is_none() {
            return Err("live native session disappeared before stop finalization".to_owned());
        }
        self.clear_native_runtime_after_drain(&fence.runtime)?;
        Ok(true)
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn native_runtime_fence(&self) -> Result<Option<NativeRuntimeFence>, String> {
        self.native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())
            .map(|runtime| runtime.clone())
    }

    /// Freeze the one capture-clock point used by both native stop events.
    /// The wall clock is always derived from the runtime anchor, never from a
    /// fresh system-clock read, so CoreAudio chronology remains coherent.
    fn capture_native_stop_point(
        &self,
        runtime: &DispatcherRuntime,
        observed_monotonic_ns: u64,
    ) -> Result<CapturePoint, String> {
        let point = native_capture_point_at(runtime, observed_monotonic_ns)?;
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        match fence.phase {
            NativeRuntimePhase::Closing | NativeRuntimePhase::Drained => {
                if let Some(existing) = fence.capture_stop_point.as_ref() {
                    return Ok(existing.clone());
                }
                fence.capture_stop_point = Some(point.clone());
                Ok(point)
            }
            NativeRuntimePhase::Active => Err(
                "native capture stop point cannot be recorded before shutdown begins".to_owned(),
            ),
            NativeRuntimePhase::Handoff => Err(
                "native capture stop point cannot be recorded during runtime handoff".to_owned(),
            ),
        }
    }

    fn native_capture_stop_point(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<CapturePoint, String> {
        let native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_ref()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Drained {
            return Err(
                "native capture stop point is unavailable before dispatcher drain".to_owned(),
            );
        }
        fence
            .capture_stop_point
            .clone()
            .ok_or_else(|| "native capture stop point is missing after input stop".to_owned())
    }

    fn record_native_capture_input_stopped(
        &self,
        runtime: &DispatcherRuntime,
        capture_stop_point: &CapturePoint,
    ) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let payload = serde_json::json!({
            "sessionId": runtime.session_id,
            "runtimeId": runtime.id.as_uuid(),
            "captureSegmentId": runtime.capture_segment_id,
        });
        let event = trail
            .next_event(
                Some(runtime.session_id),
                None,
                AuditKind::CaptureInputStopped,
                capture_stop_point.monotonic_ns,
                capture_stop_point.wall_clock,
                &payload,
            )
            .map_err(|error| format!("could not serialize capture input stop: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append(&event)
            .map_err(|error| format!("could not persist capture input stop: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified capture input stop event".to_owned());
        }
        Ok(())
    }

    /// Atomically records a native inference terminal outcome and its audit
    /// event. The captured range remains distinct from a physical capture gap.
    pub(crate) fn record_inference_gap(&self, gap: &InferenceGap) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let existing = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .lookup_inference_gap_with_audit(gap.id)
            .map_err(|error| format!("could not query persisted inference gap: {error}"))?;
        if let Some(existing) = existing {
            if existing.gap != *gap {
                return Err(
                    "inference gap ID is already bound to a different immutable payload".to_owned(),
                );
            }
            if let Some(in_memory) = trail
                .events()
                .iter()
                .find(|event| event.id == existing.audit_event.id)
            {
                if in_memory == &existing.audit_event {
                    return Ok(());
                }
                return Err(
                    "inference gap ID is bound to an audit event that differs from the in-memory trail"
                        .to_owned(),
                );
            }
            if !trail.append_event(existing.audit_event) {
                return Err(
                    "could not restore the persisted inference gap audit event into the in-memory trail"
                        .to_owned(),
                );
            }
            return Ok(());
        }

        let event = trail
            .next_event(
                Some(gap.session_id),
                gap.job_id,
                AuditKind::InferenceGapRecorded,
                gap.ended_at.monotonic_ns,
                gap.ended_at.wall_clock,
                gap,
            )
            .map_err(|error| format!("could not serialize inference gap: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append_inference_gap_with_audit(&event, gap)
            .map_err(|error| format!("could not persist inference gap: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified inference gap event".to_owned());
        }
        Ok(())
    }

    /// Register the dispatcher generation that is allowed to write native
    /// outcomes for the active recording session.
    #[cfg(test)]
    pub(crate) fn activate_native_runtime(&self, runtime: DispatcherRuntime) -> Result<(), String> {
        validate_dispatcher_runtime(&runtime)?;
        let session = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?
            .get(&runtime.session_id)
            .cloned()
            .ok_or_else(|| "native dispatcher references an unknown session".to_owned())?;
        if !matches!(session.state, crate::domain::SessionState::Recording) {
            return Err("native dispatcher requires an active recording session".to_owned());
        }

        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        if let Some(existing) = native_runtime.as_ref() {
            if existing.runtime == runtime && existing.phase == NativeRuntimePhase::Active {
                return Ok(());
            }
            return Err("a different native dispatcher runtime is already registered".to_owned());
        }
        *native_runtime = Some(NativeRuntimeFence {
            runtime,
            phase: NativeRuntimePhase::Active,
            capture_input_stopped_audited: false,
            capture_stop_point: None,
        });
        Ok(())
    }

    /// Publish a staged native session only after the CPAL stream and its
    /// dispatcher both completed the arm handoff. The start/stop lifecycle
    /// gate is held by the caller while this runs.
    fn publish_native_capture_recording(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<CaptureSession, String> {
        let native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_ref()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Active {
            return Err("native dispatcher cannot publish after shutdown begins".to_owned());
        }

        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?;
        let session = sessions
            .get_mut(&runtime.session_id)
            .ok_or_else(|| "native dispatcher references an unknown session".to_owned())?;
        session.publish_recording()?;
        Ok(session.clone())
    }

    /// Fence an active dispatcher before its capture producer is stopped.
    ///
    /// Closing outcomes are still accepted so the final drain can preserve a
    /// transcript or a durable gap before the session stop event is written.
    pub(crate) fn begin_native_runtime_shutdown(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        match fence.phase {
            NativeRuntimePhase::Active => {
                fence.phase = NativeRuntimePhase::Closing;
                Ok(())
            }
            NativeRuntimePhase::Closing => Ok(()),
            NativeRuntimePhase::Handoff | NativeRuntimePhase::Drained => Err(
                "native dispatcher shutdown has already moved beyond its closing phase".to_owned(),
            ),
        }
    }

    fn begin_native_runtime_handoff(&self, runtime: &DispatcherRuntime) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Closing {
            return Err("native dispatcher cannot be handed off before closing".to_owned());
        }
        fence.phase = NativeRuntimePhase::Handoff;
        Ok(())
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn restore_native_runtime_closing(&self, runtime: &DispatcherRuntime) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Handoff {
            return Err("native dispatcher handoff is not active".to_owned());
        }
        fence.phase = NativeRuntimePhase::Closing;
        Ok(())
    }

    fn mark_native_runtime_drained(&self, runtime: &DispatcherRuntime) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Handoff {
            return Err("native dispatcher cannot be marked drained before handoff".to_owned());
        }
        fence.phase = NativeRuntimePhase::Drained;
        Ok(())
    }

    /// Remove the generation fence only after its dispatcher has terminally
    /// drained every outcome.
    pub(crate) fn clear_native_runtime_after_drain(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_ref()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Drained {
            return Err("cannot clear a native dispatcher before its drain completes".to_owned());
        }
        *native_runtime = None;
        Ok(())
    }

    fn native_runtime_context(&self) -> Result<Option<DispatcherRuntime>, String> {
        self.native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())
            .map(|runtime| {
                runtime.as_ref().and_then(|fence| {
                    matches!(
                        fence.phase,
                        NativeRuntimePhase::Active | NativeRuntimePhase::Closing
                    )
                    .then(|| fence.runtime.clone())
                })
            })
    }

    fn native_runtime_needs_capture_input_stop_event(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<bool, String> {
        let native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_ref()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Drained {
            return Err("native capture input cannot stop before dispatcher drain".to_owned());
        }
        Ok(!fence.capture_input_stopped_audited)
    }

    fn mark_native_capture_input_stopped(&self, runtime: &DispatcherRuntime) -> Result<(), String> {
        let mut native_runtime = self
            .native_runtime
            .lock()
            .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
        let fence = native_runtime
            .as_mut()
            .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
        if fence.runtime != *runtime {
            return Err("native dispatcher runtime does not match the active fence".to_owned());
        }
        if fence.phase != NativeRuntimePhase::Drained {
            return Err(
                "native capture input cannot be marked stopped before dispatcher drain".to_owned(),
            );
        }
        fence.capture_input_stopped_audited = true;
        Ok(())
    }

    /// Persist one terminal native outcome under its runtime and segment
    /// fence. This intentionally has no knowledge of the capture-service
    /// mutex: callers claim an owned outcome, call this method, then commit or
    /// abort the lease after the SQLite transaction succeeds or fails.
    pub(crate) fn persist_native_outcome(
        &self,
        runtime: &DispatcherRuntime,
        outcome: &AsrOutcome,
    ) -> Result<Vec<TranscriptSpan>, String> {
        let session = self.active_session_for_native_runtime(runtime)?;

        match outcome {
            AsrOutcome::Gap(gap) => {
                validate_native_inference_gap(runtime, gap)?;
                self.record_inference_gap(gap)?;
                Ok(Vec::new())
            }
            AsrOutcome::Response { job, response } => {
                validate_native_asr_job(runtime, job)?;
                match validate_native_asr_response(job, response) {
                    Ok(true) => {
                        let projections = self.append_local_asr_response_with_capture_anchor(
                            &session,
                            response.clone(),
                            &runtime.capture_anchor,
                        )?;
                        if !projections.is_empty() {
                            self.publish_native_final_transcript_projection(session.id);
                        }
                        Ok(projections)
                    }
                    // A completed worker result that contains no final, or
                    // whose emissions violate the local contract, has no
                    // recoverable transcript. Account for the captured range
                    // once instead of making its result lease retry forever.
                    Ok(false) | Err(_) => self.record_native_response_failure_gap(runtime, job),
                }
            }
        }
    }

    fn record_native_response_failure_gap(
        &self,
        runtime: &DispatcherRuntime,
        job: &AsrJobMetadata,
    ) -> Result<Vec<TranscriptSpan>, String> {
        let gap = InferenceGap::new(
            // A completed job has a globally unique UUID. Reusing it as the
            // derived gap identity makes a retry after a lost lease-commit
            // acknowledgement converge on the same terminal evidence.
            job.id,
            runtime.session_id,
            runtime.id.as_uuid(),
            runtime.capture_segment_id,
            Some(job.id),
            job.started_at.clone(),
            job.ended_at.clone(),
            InferenceGapStage::Worker,
            InferenceGapReason::EngineFailed,
        )
        .map_err(|error| {
            format!("could not create terminal inference gap for failed ASR output: {error}")
        })?;
        self.record_inference_gap(&gap)?;
        Ok(Vec::new())
    }

    fn active_session_for_native_runtime(
        &self,
        runtime: &DispatcherRuntime,
    ) -> Result<CaptureSession, String> {
        validate_dispatcher_runtime(runtime)?;
        {
            let native_runtime = self
                .native_runtime
                .lock()
                .map_err(|_| "native runtime fence lock poisoned".to_owned())?;
            let fence = native_runtime
                .as_ref()
                .ok_or_else(|| "no native dispatcher runtime is registered".to_owned())?;
            if fence.runtime != *runtime {
                return Err("native dispatcher runtime does not match the active fence".to_owned());
            }
            if matches!(
                fence.phase,
                NativeRuntimePhase::Handoff | NativeRuntimePhase::Drained
            ) {
                return Err(
                    "native dispatcher outcome arrived after runtime drain began".to_owned(),
                );
            }
        }

        let session = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?
            .get(&runtime.session_id)
            .cloned()
            .ok_or_else(|| "native dispatcher references an unknown session".to_owned())?;
        if !matches!(
            session.state,
            crate::domain::SessionState::Starting | crate::domain::SessionState::Recording
        ) {
            return Err("native dispatcher outcome arrived after its session stopped".to_owned());
        }
        Ok(session)
    }

    #[cfg(any(test, debug_assertions))]
    fn start_session_with_active(
        &self,
        reject_active_session: bool,
    ) -> Result<CaptureSession, String> {
        self.start_session_at(
            crate::audio::CapturePoint {
                monotonic_ns: self.monotonic_ns(),
                wall_clock: Utc::now(),
            },
            reject_active_session,
        )
    }

    #[cfg(any(test, debug_assertions))]
    fn start_session_at(
        &self,
        point: crate::audio::CapturePoint,
        reject_active_session: bool,
    ) -> Result<CaptureSession, String> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?;

        if let Some(active) = sessions.values().find(|session| {
            matches!(
                session.state,
                crate::domain::SessionState::Starting | crate::domain::SessionState::Recording
            )
        }) {
            if reject_active_session {
                return Err(
                    "stop the active recording session before starting a development mock"
                        .to_owned(),
                );
            }
            return Ok(active.clone());
        }

        let session = CaptureSession::begin(point.monotonic_ns, point.wall_clock);
        self.record_audit(AuditKind::SessionStarted, point.monotonic_ns, &session)?;
        self.timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?
            .entry(session.id)
            .or_default();
        sessions.insert(session.id, session.clone());
        Ok(session)
    }

    #[cfg(any(test, debug_assertions))]
    pub fn start_development_mock_session(&self) -> Result<CaptureSession, String> {
        let mut development_mock = self
            .development_mock
            .lock()
            .map_err(|_| "development mock state lock poisoned".to_owned())?;
        if development_mock.is_some() {
            return Err("a development mock session is already active".to_owned());
        }

        let session = self.start_session_with_active(true)?;
        *development_mock = Some(DevelopmentMockRunner::new(&session)?);
        Ok(session)
    }

    #[cfg(any(test, debug_assertions))]
    pub fn advance_development_mock(
        &self,
        packet_count: usize,
    ) -> Result<DevelopmentMockProgress, String> {
        let active_session_id = self
            .recording_session_id()?
            .ok_or_else(|| "no recording session is active".to_owned())?;
        let mut persisted_spans = self.flush_development_mock_responses(active_session_id)?;
        let mut progress = {
            let mut development_mock = self
                .development_mock
                .lock()
                .map_err(|_| "development mock state lock poisoned".to_owned())?;
            let runner = development_mock
                .as_mut()
                .ok_or_else(|| "no development mock session is active".to_owned())?;
            if runner.session_id() != active_session_id {
                return Err("development mock does not match the active session".to_owned());
            }
            runner.advance(packet_count)?
        };
        persisted_spans.extend(self.flush_development_mock_responses(active_session_id)?);
        progress.spans = persisted_spans;
        Ok(progress)
    }

    pub fn stop_session(&self) -> Result<Option<CaptureSession>, String> {
        #[cfg(all(target_os = "macos", not(test)))]
        let _native_lifecycle = self
            .native_capture_lifecycle
            .lock()
            .map_err(|_| "native capture lifecycle lock poisoned".to_owned())?;

        #[cfg(all(target_os = "macos", not(test)))]
        let native_capture_stop_point = self.stop_native_capture_before_session_stop()?;

        #[cfg(not(all(target_os = "macos", not(test))))]
        let native_capture_stop_point: Option<NativeCaptureStop> = None;

        #[cfg(any(test, debug_assertions))]
        if let Some(session) = self.active_recording_session()? {
            // A development final must be durable before the stop event is
            // sealed, otherwise an evidence trail could claim a stopped
            // session before its last local inference result exists.
            self.flush_development_mock_responses(session.id)?;
            self.stop_development_mock(session.id)?;
            self.flush_development_mock_responses(session.id)?;
            self.finish_development_mock(session.id)?;
        }

        if let Some(session) = self.active_live_session()? {
            self.clear_local_asr_session(session.id)?;
        }

        let stop_point = native_capture_stop_point
            .as_ref()
            .map(|native_stop| native_stop.point.clone())
            .unwrap_or_else(|| CapturePoint {
                monotonic_ns: self.monotonic_ns(),
                wall_clock: Utc::now(),
            });
        let stopped = self.finish_capture_session_at(stop_point)?;

        // Keep the Drained fence, including its captured stop point, until
        // SessionStopped is durable. A retry after an audit failure then
        // reuses the exact same chronology without duplicating input stop.
        if let Some(native_stop) = native_capture_stop_point.as_ref() {
            self.clear_native_runtime_after_drain(&native_stop.runtime)?;
        }

        Ok(stopped)
    }

    fn finish_capture_session_at(
        &self,
        stop_point: CapturePoint,
    ) -> Result<Option<CaptureSession>, String> {
        let stopped = {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|_| "session state lock poisoned".to_owned())?;
            let Some(session) = sessions.values_mut().find(|session| {
                matches!(
                    session.state,
                    crate::domain::SessionState::Starting | crate::domain::SessionState::Recording
                )
            }) else {
                return Ok(None);
            };

            let mut stopped = session.clone();
            stopped.stop(stop_point.wall_clock);
            self.record_audit_at(
                AuditKind::SessionStopped,
                stop_point.monotonic_ns,
                stop_point.wall_clock,
                &stopped,
            )?;
            *session = stopped.clone();
            stopped
        };

        Ok(Some(stopped))
    }

    pub fn list_timeline(&self, session_id: Option<Uuid>) -> Result<Vec<TranscriptSpan>, String> {
        let timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut spans = match session_id {
            Some(session_id) => timelines.get(&session_id).cloned().unwrap_or_default(),
            // There is no live session after a cold restart. Show the most
            // recently recorded local session rather than mixing unrelated
            // archives into a view labelled as the current conversation.
            None => timelines
                .iter()
                .max_by_key(|(_, timeline)| {
                    timeline
                        .iter()
                        .filter_map(|span| span.wall_clock_start)
                        .max()
                })
                .map_or_else(Vec::new, |(_, timeline)| timeline.clone()),
        };
        spans.sort_by_key(|span| (span.capture_start_ns, span.revision));
        Ok(spans)
    }

    /// Lists the compact, session-scoped anonymous speaker catalog. With no
    /// explicit session, prefer the active capture and otherwise mirror the
    /// timeline's most-recent-session behavior.
    pub fn list_speaker_clusters(
        &self,
        session_id: Option<Uuid>,
    ) -> Result<Vec<SpeakerCluster>, String> {
        let Some(session_id) = self.speaker_catalog_session_id(session_id)? else {
            return Ok(Vec::new());
        };
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))
    }

    /// Creates the next locally generated anonymous speaker catalog entry.
    /// The catalog never records an inferred identity; the initial label is
    /// generated from its session-scoped ordinal.
    pub fn create_speaker_cluster(
        &self,
        session_id: Uuid,
    ) -> Result<SpeakerOperationResult, String> {
        // Keep this lock order aligned with speaker reassignment and final
        // transcript persistence: timelines -> audit trail -> audit store.
        let timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        if !timelines.contains_key(&session_id) {
            return Err("speaker cluster session does not exist".to_owned());
        }
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let mut audit_store = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?;
        let current_clusters = audit_store
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))?;
        let ordinal = u32::try_from(current_clusters.len())
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| "speaker cluster ordinal overflowed".to_owned())?;
        let cluster = SpeakerClusterRecord::new(session_id, ordinal)
            .map_err(|error| format!("could not create speaker cluster: {error}"))?;
        let initial_label = SpeakerClusterLabelRevision::initial_generated(&cluster)
            .map_err(|error| format!("could not create initial speaker label: {error}"))?;
        let payload =
            SpeakerClusterCreatedAuditPayload::new(cluster.clone(), initial_label.clone())
                .map_err(|error| {
                    format!("could not prepare speaker cluster audit payload: {error}")
                })?;
        let event = trail
            .next_event(
                Some(session_id),
                None,
                AuditKind::SpeakerClusterCreated,
                self.monotonic_ns(),
                Utc::now(),
                &payload,
            )
            .map_err(|error| format!("could not serialize speaker cluster: {error}"))?;
        audit_store
            .append_speaker_cluster_with_audit(&event, &cluster, &initial_label)
            .map_err(|error| format!("could not persist speaker cluster: {error}"))?;
        assert!(
            trail.append_event(event),
            "an audit event generated while holding the trail lock must append"
        );
        let clusters = audit_store
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))?;

        Ok(SpeakerOperationResult {
            clusters,
            updated_spans: Vec::new(),
        })
    }

    /// Appends a user-entered presentation-label revision after checking the
    /// caller's current label revision. Labels remain anonymous metadata.
    pub fn rename_speaker_cluster(
        &self,
        session_id: Uuid,
        cluster_id: String,
        expected_label_revision: u32,
        label: String,
    ) -> Result<SpeakerOperationResult, String> {
        let _timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let mut audit_store = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?;
        let cluster = audit_store
            .get_speaker_cluster_record(&cluster_id)
            .map_err(|error| format!("could not load speaker cluster: {error}"))?
            .ok_or_else(|| "speaker cluster does not exist".to_owned())?;
        if cluster.session_id != session_id {
            return Err("speaker cluster does not belong to this session".to_owned());
        }
        let previous = audit_store
            .get_latest_speaker_cluster_label_revision(&cluster_id)
            .map_err(|error| format!("could not load speaker label revision: {error}"))?
            .ok_or_else(|| "speaker cluster has no durable label revision".to_owned())?;
        if previous.revision != expected_label_revision {
            return Err("speaker label has changed; refresh before renaming".to_owned());
        }
        let revision = SpeakerClusterLabelRevision::revision_of(&previous, label)
            .map_err(|error| format!("invalid speaker label: {error}"))?;
        if revision.label == previous.label {
            return Err("speaker cluster already uses that label".to_owned());
        }
        let event = trail
            .next_event(
                Some(session_id),
                revision.parent_revision_id,
                AuditKind::SpeakerClusterLabelRevisionRecorded,
                self.monotonic_ns(),
                Utc::now(),
                &revision,
            )
            .map_err(|error| format!("could not serialize speaker label revision: {error}"))?;
        audit_store
            .append_speaker_cluster_label_revision_with_audit(&event, &revision)
            .map_err(|error| format!("could not persist speaker label revision: {error}"))?;
        assert!(
            trail.append_event(event),
            "an audit event generated while holding the trail lock must append"
        );
        let clusters = audit_store
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))?;

        Ok(SpeakerOperationResult {
            clusters,
            updated_spans: Vec::new(),
        })
    }

    /// Appends exactly one speaker-only final revision for the durable current
    /// head of a logical span. No caller-provided text or capture metadata can
    /// enter this path.
    pub fn reassign_transcript_speaker(
        &self,
        session_id: Uuid,
        logical_span_id: Uuid,
        expected_revision: u32,
        target_cluster_id: Option<String>,
    ) -> Result<SpeakerOperationResult, String> {
        let mut timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let mut audit_store = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?;
        let revisions = audit_store
            .list_transcript_revisions(session_id)
            .map_err(|error| format!("could not load durable transcript revisions: {error}"))?;
        let current = durable_current_final_transcript_head(&revisions, logical_span_id)
            .cloned()
            .ok_or_else(|| "final transcript span does not exist in this session".to_owned())?;
        if current.revision != expected_revision {
            return Err("transcript span has changed; refresh before reassigning".to_owned());
        }

        let current_clusters = audit_store
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))?;
        if let Some(target_cluster_id) = target_cluster_id.as_deref() {
            let target = current_clusters
                .iter()
                .find(|cluster| cluster.id == target_cluster_id)
                .ok_or_else(|| {
                    "target speaker cluster does not exist in this session".to_owned()
                })?;
            if target.merged_into_cluster_id.is_some() || target.canonical_cluster_id != target.id {
                return Err(
                    "target speaker cluster is merged and cannot receive assignments".to_owned(),
                );
            }
        }
        if current.speaker_cluster_id == target_cluster_id {
            return Err("speaker assignment already matches the requested target".to_owned());
        }

        let revision = TranscriptRevision::revision_of(
            &current,
            current.timing(),
            target_cluster_id,
            current.text.clone(),
            true,
            TranscriptSource::UserEdited,
            current.model.clone(),
            current.confidence,
        )
        .map_err(|error| format!("could not prepare speaker reassignment: {error}"))?;
        let event = trail
            .next_event(
                Some(session_id),
                revision.parent_revision_id,
                AuditKind::TranscriptSpeakerReassigned,
                self.monotonic_ns(),
                Utc::now(),
                &revision,
            )
            .map_err(|error| format!("could not serialize speaker reassignment: {error}"))?;
        audit_store
            .append_transcript_speaker_reassignment_with_audit(&event, &revision)
            .map_err(|error| format!("could not persist speaker reassignment: {error}"))?;
        assert!(
            trail.append_event(event),
            "an audit event generated while holding the trail lock must append"
        );

        // SQLite has committed both immutable records before the WebView's
        // in-memory projection is allowed to observe the new assignment.
        Self::upsert_timeline_projection(&mut timelines, transcript_revision_projection(&revision));
        let clusters = audit_store
            .list_speaker_clusters(session_id)
            .map_err(|error| format!("could not load speaker catalog: {error}"))?;

        Ok(SpeakerOperationResult {
            clusters,
            updated_spans: vec![SpeakerSpanRef {
                id: logical_span_id,
                revision: revision.revision,
            }],
        })
    }

    fn speaker_catalog_session_id(
        &self,
        requested_session_id: Option<Uuid>,
    ) -> Result<Option<Uuid>, String> {
        if requested_session_id.is_some() {
            return Ok(requested_session_id);
        }
        if let Some(session) = self.active_live_session()? {
            return Ok(Some(session.id));
        }

        self.timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())
            .map(|timelines| {
                timelines
                    .iter()
                    .max_by_key(|(_, timeline)| {
                        timeline
                            .iter()
                            .filter_map(|span| span.wall_clock_start)
                            .max()
                    })
                    .map(|(session_id, _)| *session_id)
            })
    }

    pub fn append_transcript(&self, span: TranscriptSpan) -> Result<(), String> {
        let monotonic_ns = span.capture_end_ns;
        self.timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?
            .entry(span.session_id)
            .or_default()
            .push(span.clone());
        self.record_audit(AuditKind::TranscriptRecorded, monotonic_ns, &span)
    }

    /// Persist a final non-ASR correction or synthetic revision before exposing
    /// its compact timeline projection. Native ASR finals must use
    /// [`Self::append_local_asr_response`] so their replay binding is committed
    /// with the transcript audit event.
    pub fn append_final_transcript_revision(
        &self,
        revision: TranscriptRevision,
    ) -> Result<(), String> {
        if !revision.is_final {
            return Err(
                "only final transcript revisions may be persisted for Agent context".to_owned(),
            );
        }
        if revision.source == TranscriptSource::LocalInference {
            return Err(
                "local inference transcript revisions must be persisted from an ASR response"
                    .to_owned(),
            );
        }
        revision
            .validate()
            .map_err(|error| format!("invalid final transcript revision: {error}"))?;

        let projection = transcript_revision_projection(&revision);
        // Acquire every fallible in-memory resource before the SQLite
        // transaction. Once it commits, callers must not receive a retryable
        // error for this final revision.
        let mut timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(
                Some(revision.session_id),
                revision.parent_revision_id,
                AuditKind::TranscriptRevisionRecorded,
                revision.capture_end_ns,
                revision.wall_clock_end,
                &revision,
            )
            .map_err(|error| format!("could not serialize transcript revision: {error}"))?;
        {
            let mut audit_store = self
                .audit_store
                .lock()
                .map_err(|_| "audit store lock poisoned".to_owned())?;
            audit_store
                .append_transcript_revision_with_audit(&event, &revision)
                .map_err(|error| format!("could not persist transcript revision: {error}"))?;
        }
        assert!(
            trail.append_event(event),
            "an audit event generated while holding the trail lock must append"
        );
        drop(trail);

        Self::upsert_timeline_projection(&mut timelines, projection);
        Ok(())
    }

    /// Persists a final native ASR emission with a durable replay key. A
    /// matching payload is an already-committed replay; a different payload
    /// for the same source identity is an integrity error rather than a new
    /// transcript revision.
    fn append_final_asr_transcript_revision(
        &self,
        revision: TranscriptRevision,
        key: &crate::inference::AsrFinalIdempotencyKey,
        emission_payload_sha256: &str,
    ) -> Result<Option<TranscriptSpan>, String> {
        if !revision.is_final {
            return Err(
                "only final transcript revisions may be persisted for Agent context".to_owned(),
            );
        }
        revision
            .validate()
            .map_err(|error| format!("invalid final transcript revision: {error}"))?;
        let idempotency = AsrFinalIdempotencyBinding::new(key, &revision, emission_payload_sha256)
            .map_err(|error| format!("invalid local ASR idempotency binding: {error}"))?;

        let projection = transcript_revision_projection(&revision);
        // Keep the same lock order as other audit writes. Checking for an
        // existing key before generating the in-memory audit event means a
        // harmless replay cannot advance the hash chain.
        let mut timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let mut audit_store = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?;
        if let Some(existing) = audit_store
            .lookup_asr_final_idempotency(key)
            .map_err(|error| format!("could not query local ASR idempotency: {error}"))?
        {
            if existing.emission_payload_sha256 == idempotency.emission_payload_sha256 {
                return Ok(None);
            }
            return Err(
                "a local ASR final conflicts with its durable idempotency record".to_owned(),
            );
        }

        let audit_payload = AsrFinalAuditPayload::new(&revision, &idempotency);
        let event = trail
            .next_event(
                Some(revision.session_id),
                revision.parent_revision_id,
                AuditKind::TranscriptRevisionRecorded,
                revision.capture_end_ns,
                revision.wall_clock_end,
                &audit_payload,
            )
            .map_err(|error| format!("could not serialize transcript revision: {error}"))?;
        audit_store
            .append_asr_final_transcript_revision_with_audit(&event, &revision, &idempotency)
            .map_err(|error| format!("could not persist transcript revision: {error}"))?;
        assert!(
            trail.append_event(event),
            "an audit event generated while holding the trail lock must append"
        );
        drop(audit_store);
        drop(trail);

        Self::upsert_timeline_projection(&mut timelines, projection.clone());
        Ok(Some(projection))
    }

    fn upsert_timeline_projection(
        timelines: &mut BTreeMap<Uuid, Vec<TranscriptSpan>>,
        span: TranscriptSpan,
    ) {
        let timeline = timelines.entry(span.session_id).or_default();
        if let Some(previous) = timeline.iter_mut().find(|previous| previous.id == span.id) {
            if span.revision >= previous.revision {
                *previous = span;
            }
        } else {
            timeline.push(span);
        }
    }

    /// Applies one native ASR response to its active local capture session.
    /// Partial output never reaches SQLite, FTS, the timeline, or Agent
    /// context; only final emissions become first durable revisions.
    pub fn append_local_asr_response(
        &self,
        session_id: Uuid,
        response: AsrResponse,
    ) -> Result<Vec<TranscriptSpan>, String> {
        let session = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?
            .get(&session_id)
            .cloned()
            .ok_or_else(|| "ASR response references an unknown local session".to_owned())?;

        let capture_anchor = CapturePoint {
            monotonic_ns: session.started_monotonic_ns,
            wall_clock: session.started_at,
        };
        self.append_local_asr_response_with_capture_anchor(&session, response, &capture_anchor)
    }

    fn append_local_asr_response_with_capture_anchor(
        &self,
        session: &CaptureSession,
        response: AsrResponse,
        capture_anchor: &CapturePoint,
    ) -> Result<Vec<TranscriptSpan>, String> {
        if capture_anchor.monotonic_ns < session.started_monotonic_ns {
            return Err("ASR capture anchor begins before its session".to_owned());
        }

        let mut mapper = self
            .inference_mapper
            .lock()
            .map_err(|_| "inference mapper lock poisoned".to_owned())?;
        let mut projections = Vec::new();
        for emission in response.emissions {
            let mapped = mapper
                .map(session.id, emission)
                .map_err(|error| format!("could not map local ASR emission: {error}"))?;
            let MappedTranscriptEmission::Final(final_emission) = mapped else {
                continue;
            };

            let persisted = (|| {
                let revision = transcript_revision_from_final_emission(
                    session,
                    &final_emission.emission,
                    final_emission.logical_span_id,
                    capture_anchor,
                )?;
                self.append_final_asr_transcript_revision(
                    revision,
                    &final_emission.idempotency_key(),
                    &final_emission.idempotency_payload_sha256(),
                )
            })();
            match persisted {
                Ok(Some(projection)) => {
                    // Keep the reservation pending until SQLite and its audit
                    // event have both committed, so concurrent/replayed finals
                    // cannot create a second durable revision.
                    mapper.commit_final(&final_emission);
                    projections.push(projection);
                }
                Ok(None) => {
                    // SQLite already owns this exact final; releasing the
                    // mapper reservation prevents a replay from retaining its
                    // text in memory.
                    mapper.commit_final(&final_emission);
                }
                Err(error) => {
                    mapper.abort_final(&final_emission).map_err(|abort_error| {
                        format!("{error}; could not release final ASR reservation: {abort_error}")
                    })?;
                    return Err(error);
                }
            }
        }
        Ok(projections)
    }

    fn clear_local_asr_session(&self, session_id: Uuid) -> Result<(), String> {
        self.inference_mapper
            .lock()
            .map_err(|_| "inference mapper lock poisoned".to_owned())?
            .clear_session(session_id)
            .map_err(|error| format!("could not finish local ASR session: {error}"))
    }

    /// Returns the newest in-process native final-transcript notification.
    /// It is intentionally not reconstructed from stored transcript content:
    /// a new WebView performs its normal initial timeline load instead.
    pub fn final_transcript_projection(&self) -> Option<FinalTranscriptProjection> {
        self.final_transcript_projection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .latest
            .clone()
    }

    fn publish_native_final_transcript_projection(&self, session_id: Uuid) {
        let mut state = self
            .final_transcript_projection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.revision = state
            .revision
            .checked_add(1)
            .expect("final transcript projection revision overflowed");
        state.latest = Some(FinalTranscriptProjection {
            session_id,
            revision: state.revision,
        });
    }

    pub fn list_local_models(&self) -> Result<Vec<RegisteredModel>, String> {
        let registry = self
            .model_registry
            .lock()
            .map_err(|_| "model registry lock poisoned".to_owned())?;
        Ok(registry.models().cloned().collect())
    }

    /// Returns the deliberately selected ASR model for this application run.
    pub fn active_local_asr_profile(&self) -> Result<Option<ActiveLocalAsrProfile>, String> {
        self.active_local_asr_profile
            .lock()
            .map_err(|_| "active local ASR profile lock poisoned".to_owned())
            .map(|profile| profile.clone())
    }

    /// Select a compatible, already-imported local Whisper model. Importing a
    /// file never selects it automatically; changing this choice during an
    /// active capture would make its provenance ambiguous and is rejected.
    pub fn select_active_local_asr_model(
        &self,
        model_id: Uuid,
    ) -> Result<ActiveLocalAsrProfile, String> {
        // Share the native start/stop gate so a model cannot change while a
        // microphone start is loading and binding its selected artifact.
        #[cfg(target_os = "macos")]
        let _native_lifecycle = self
            .native_capture_lifecycle
            .lock()
            .map_err(|_| "native capture lifecycle lock poisoned".to_owned())?;

        if self.active_live_session()?.is_some() {
            return Err("cannot change the local transcription model while recording".to_owned());
        }

        let model = self
            .model_registry
            .lock()
            .map_err(|_| "model registry lock poisoned".to_owned())?
            .get(model_id)
            .cloned()
            .ok_or_else(|| "the selected local transcription model is not registered".to_owned())?;
        validate_active_local_asr_model(&model)?;

        let profile = ActiveLocalAsrProfile { model_id };
        *self
            .active_local_asr_profile
            .lock()
            .map_err(|_| "active local ASR profile lock poisoned".to_owned())? =
            Some(profile.clone());
        Ok(profile)
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn build_active_local_inference_engines(&self) -> Result<NativeInferenceEngines, String> {
        let profile = self.active_local_asr_profile().and_then(|profile| {
            profile.ok_or_else(|| {
                "select a compatible local Whisper model before starting microphone recording"
                    .to_owned()
            })
        })?;
        let artifact = self
            .model_registry
            .lock()
            .map_err(|_| "model registry lock poisoned".to_owned())?
            .verified_artifact(profile.model_id)
            .map_err(|error| format!("could not verify the selected local ASR model: {error}"))?;
        validate_active_local_asr_model(artifact.model())?;
        let asr = WhisperCppAsrEngine::from_registered_artifact(artifact)
            .map_err(|error| format!("could not load the selected local ASR model: {error}"))?;
        Ok(NativeInferenceEngines::new(WebRtcVad::new(), asr))
    }

    /// Imports a user-selected model into application-managed local storage.
    ///
    /// The file copy is rolled back if the accompanying model/audit database
    /// transaction cannot commit. This code never creates a network client.
    pub fn import_local_model(
        &self,
        request: ModelImportRequest,
    ) -> Result<RegisteredModel, String> {
        let model_root = self.model_root.as_ref().ok_or_else(|| {
            "local model import requires an absolute application data path".to_owned()
        })?;
        let registration = self
            .model_registry
            .lock()
            .map_err(|_| "model registry lock poisoned".to_owned())?
            .import(model_root, request)
            .map_err(|error| format!("could not import local model: {error}"))?;

        if let Err(error) = self.record_local_model_imported(&registration) {
            let mut registry = self
                .model_registry
                .lock()
                .map_err(|_| "model registry lock poisoned during rollback".to_owned())?;
            let rolled_back = registry.rollback_registration(registration.id);
            if let Some(rolled_back) = rolled_back {
                if let Err(cleanup_error) = registry.remove_managed_artifact(&rolled_back) {
                    return Err(format!(
                        "{error}; could not remove unregistered local model copy: {cleanup_error}"
                    ));
                }
            }
            return Err(error);
        }

        Ok(registration)
    }

    #[cfg(any(test, debug_assertions))]
    fn recording_session_id(&self) -> Result<Option<Uuid>, String> {
        self.sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())
            .map(|sessions| {
                sessions
                    .values()
                    .find(|session| matches!(session.state, crate::domain::SessionState::Recording))
                    .map(|session| session.id)
            })
    }

    #[cfg(any(test, debug_assertions))]
    fn flush_development_mock_responses(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<TranscriptSpan>, String> {
        let mut projections = Vec::new();
        loop {
            // Keep the mock delivery claimed until its synchronous native
            // persistence either commits or aborts. Releasing this lock in
            // between could leave the queue head permanently in-flight when
            // acknowledgement itself fails.
            let mut development_mock = self
                .development_mock
                .lock()
                .map_err(|_| "development mock state lock poisoned".to_owned())?;
            let Some(runner) = development_mock.as_mut() else {
                return Ok(projections);
            };
            if runner.session_id() != session_id {
                return Ok(projections);
            }
            let delivery = runner.begin_pending_asr_delivery()?;
            let Some(delivery) = delivery else {
                return Ok(projections);
            };
            let delivery_id = delivery.id;
            let persisted = self.append_local_asr_response(session_id, delivery.response);
            match persisted {
                Ok(mut next_projections) => {
                    runner.commit_pending_asr_delivery(delivery_id)?;
                    projections.append(&mut next_projections);
                }
                Err(error) => {
                    if let Err(release_error) = runner.abort_pending_asr_delivery(delivery_id) {
                        return Err(format!(
                            "{error}; could not retain the failed development mock ASR response: {release_error}"
                        ));
                    }
                    return Err(error);
                }
            }
        }
    }

    #[cfg(any(test, debug_assertions))]
    fn stop_development_mock(&self, session_id: Uuid) -> Result<(), String> {
        let mut development_mock = self
            .development_mock
            .lock()
            .map_err(|_| "development mock state lock poisoned".to_owned())?;
        let Some(runner) = development_mock.as_mut() else {
            return Ok(());
        };
        if runner.session_id() == session_id {
            runner.stop()
        } else {
            Ok(())
        }
    }

    #[cfg(any(test, debug_assertions))]
    fn finish_development_mock(&self, session_id: Uuid) -> Result<(), String> {
        let mut development_mock = self
            .development_mock
            .lock()
            .map_err(|_| "development mock state lock poisoned".to_owned())?;
        let Some(runner) = development_mock.as_ref() else {
            return Ok(());
        };
        if runner.session_id() != session_id {
            return Ok(());
        }
        if runner.has_pending_asr_responses() {
            return Err("development mock cannot stop while an ASR response is pending".to_owned());
        }
        development_mock.take();
        Ok(())
    }

    pub fn create_egress_approval(
        &self,
        tool_id: String,
        origin: String,
        data_categories: BTreeSet<DataCategory>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<EgressApproval, String> {
        let approval =
            EgressApproval::new(tool_id, origin, data_categories, Utc::now(), expires_at)
                .map_err(policy_reason_message)?;
        self.policy
            .lock()
            .map_err(|_| "policy state lock poisoned".to_owned())?
            .add_approval(approval.clone());
        self.record_audit(
            AuditKind::EgressApprovalCreated,
            self.monotonic_ns(),
            &approval,
        )?;
        Ok(approval)
    }

    pub fn revoke_egress_approval(&self, approval_id: Uuid) -> Result<bool, String> {
        let revoked = self
            .policy
            .lock()
            .map_err(|_| "policy state lock poisoned".to_owned())?
            .revoke(approval_id, Utc::now());
        if revoked {
            self.record_audit(
                AuditKind::EgressApprovalRevoked,
                self.monotonic_ns(),
                &serde_json::json!({ "approvalId": approval_id }),
            )?;
        }
        Ok(revoked)
    }

    pub fn propose_local_speech(&self) -> Result<AgentAction, String> {
        let action = AgentAction {
            id: Uuid::new_v4(),
            title: "播报本地行动摘要".to_owned(),
            detail: "仅使用本机语音输出".to_owned(),
            status: ActionStatus::Ready,
            kind: "local_speech".to_owned(),
        };
        self.actions
            .lock()
            .map_err(|_| "action state lock poisoned".to_owned())?
            .insert(0, action.clone());
        self.record_audit(AuditKind::PlanProposed, self.monotonic_ns(), &action)?;
        Ok(action)
    }

    pub fn list_actions(&self) -> Result<Vec<AgentAction>, String> {
        Ok(self
            .actions
            .lock()
            .map_err(|_| "action state lock poisoned".to_owned())?
            .clone())
    }

    pub fn evaluate_http_profile(
        &self,
        tool_id: String,
        origin: String,
        data_categories: BTreeSet<DataCategory>,
    ) -> Result<PolicyDecision, String> {
        let decision = self
            .policy
            .lock()
            .map_err(|_| "policy state lock poisoned".to_owned())?
            .evaluate(
                &EgressRequest {
                    tool_id,
                    origin,
                    data_categories,
                },
                Utc::now(),
            );
        let kind = if matches!(decision, PolicyDecision::Allowed { .. }) {
            AuditKind::ActionExecuted
        } else {
            AuditKind::ActionDenied
        };
        self.record_audit(kind, self.monotonic_ns(), &decision)?;
        Ok(decision)
    }

    pub fn audit_is_valid(&self) -> Result<bool, String> {
        let trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let audit_store = self
            .audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?;
        let durable_valid = audit_store
            .verify()
            .map_err(|error| format!("could not verify durable audit records: {error}"))?;
        Ok(trail.verify() && durable_valid)
    }

    fn monotonic_ns(&self) -> u64 {
        self.started_at
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }

    fn record_audit<T: Serialize>(
        &self,
        kind: AuditKind,
        monotonic_ns: u64,
        payload: &T,
    ) -> Result<(), String> {
        self.record_audit_at(kind, monotonic_ns, Utc::now(), payload)
    }

    fn record_audit_at<T: Serialize>(
        &self,
        kind: AuditKind,
        monotonic_ns: u64,
        wall_clock: DateTime<Utc>,
        payload: &T,
    ) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(None, None, kind, monotonic_ns, wall_clock, payload)
            .map_err(|error| format!("could not serialize audit payload: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append(&event)
            .map_err(|error| format!("could not persist audit event: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified audit event".to_owned());
        }
        Ok(())
    }

    fn record_local_model_imported(&self, model: &RegisteredModel) -> Result<(), String> {
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(
                None,
                None,
                AuditKind::LocalModelImported,
                self.monotonic_ns(),
                Utc::now(),
                model,
            )
            .map_err(|error| format!("could not serialize local model audit payload: {error}"))?;
        self.audit_store
            .lock()
            .map_err(|_| "audit store lock poisoned".to_owned())?
            .append_local_model_with_audit(&event, model)
            .map_err(|error| format!("could not persist local model import: {error}"))?;
        if !trail.append_event(event) {
            return Err("could not append verified local model import event".to_owned());
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", not(test)))]
fn with_staged_capture_cleanup(primary: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup_error) => {
            format!("{primary}; could not clean up staged microphone capture: {cleanup_error}")
        }
    }
}

fn validate_active_local_asr_model(model: &RegisteredModel) -> Result<(), String> {
    if model.model_kind != LocalModelKind::SpeechRecognition {
        return Err("the selected model is not a speech-recognition model".to_owned());
    }
    if !is_whisper_cpp_compatible_input_format(&model.input_format) {
        return Err(
            "the selected speech-recognition model is not compatible with the local Whisper runtime"
                .to_owned(),
        );
    }
    Ok(())
}

fn policy_reason_message(reason: PolicyReason) -> String {
    match reason {
        PolicyReason::EgressDisabled => "network egress is disabled for this session".to_owned(),
        PolicyReason::DeniedByDefault => "network egress is denied by default".to_owned(),
        PolicyReason::MissingToolIdentifier => {
            "an egress approval needs a tool identifier".to_owned()
        }
        PolicyReason::InvalidOrigin => "an egress approval needs a valid origin".to_owned(),
        PolicyReason::InsecureOrigin => "only HTTPS origins may be approved".to_owned(),
        PolicyReason::OriginMismatch => "the requested origin does not match approval".to_owned(),
        PolicyReason::ApprovalExpired => "the egress approval has expired".to_owned(),
        PolicyReason::ApprovalRevoked => "the egress approval was revoked".to_owned(),
        PolicyReason::DataScopeMismatch => "the request exceeds the approved data scope".to_owned(),
    }
}

fn transcript_revision_projection(revision: &TranscriptRevision) -> TranscriptSpan {
    TranscriptSpan {
        // The timeline renders the latest value for a logical span. The
        // immutable physical revision ID remains in SQLite and the audit chain.
        id: revision.logical_span_id,
        session_id: revision.session_id,
        capture_start_ns: revision.capture_start_ns,
        capture_end_ns: revision.capture_end_ns,
        wall_clock_start: Some(revision.wall_clock_start),
        speaker_cluster_id: revision.speaker_cluster_id.clone(),
        text: revision.text.clone(),
        is_final: revision.is_final,
        revision: revision.revision,
        source: revision.source.clone(),
    }
}

fn transcript_revision_from_final_emission(
    session: &CaptureSession,
    emission: &TranscriptEmission,
    logical_span_id: Uuid,
    capture_anchor: &CapturePoint,
) -> Result<TranscriptRevision, String> {
    let timing = TranscriptTiming::new(
        emission.capture_start_ns,
        emission.capture_end_ns,
        capture_anchor_wall_clock_at(capture_anchor, emission.capture_start_ns)?,
        capture_anchor_wall_clock_at(capture_anchor, emission.capture_end_ns)?,
    )?;
    let model = TranscriptModelProvenance::new(
        emission.model_provenance.provider(),
        emission.model_provenance.model_id(),
        emission.model_provenance.model_version(),
        Some(emission.model_provenance.artifact_sha256().to_owned()),
    )?;

    TranscriptRevision::original_with_id(
        logical_span_id,
        session.id,
        timing,
        None,
        emission.text.clone(),
        true,
        TranscriptSource::LocalInference,
        Some(model),
        None,
    )
}

fn capture_anchor_wall_clock_at(
    capture_anchor: &CapturePoint,
    capture_ns: u64,
) -> Result<DateTime<Utc>, String> {
    let offset_ns = capture_ns
        .checked_sub(capture_anchor.monotonic_ns)
        .ok_or_else(|| "ASR emission begins before its capture segment".to_owned())?;
    let offset_ns = i64::try_from(offset_ns)
        .map_err(|_| "ASR emission offset exceeds the supported wall-clock range".to_owned())?;
    Ok(capture_anchor.wall_clock + chrono::Duration::nanoseconds(offset_ns))
}

fn validate_dispatcher_runtime(runtime: &DispatcherRuntime) -> Result<(), String> {
    if runtime.session_id.is_nil() {
        return Err("native dispatcher session ID must not be empty".to_owned());
    }
    if runtime.capture_segment_id.is_nil() {
        return Err("native dispatcher capture segment ID must not be empty".to_owned());
    }
    if runtime.id.as_uuid().is_nil() {
        return Err("native dispatcher runtime ID must not be empty".to_owned());
    }
    Ok(())
}

fn validate_native_asr_job(
    runtime: &DispatcherRuntime,
    job: &AsrJobMetadata,
) -> Result<(), String> {
    if job.id.is_nil() {
        return Err("native ASR job ID must not be empty".to_owned());
    }
    if job.session_id != runtime.session_id {
        return Err("native ASR job session does not match its dispatcher runtime".to_owned());
    }
    if job.runtime_id != runtime.id {
        return Err("native ASR job generation does not match its dispatcher runtime".to_owned());
    }
    if job.capture_segment_id != runtime.capture_segment_id {
        return Err("native ASR job segment does not match its dispatcher runtime".to_owned());
    }
    if job.ended_at.monotonic_ns <= job.started_at.monotonic_ns {
        return Err("native ASR job capture range must not be empty or inverted".to_owned());
    }
    validate_native_capture_range(runtime, &job.started_at, &job.ended_at, "native ASR job")
}

fn validate_native_asr_response(
    job: &AsrJobMetadata,
    response: &AsrResponse,
) -> Result<bool, String> {
    if response.emissions.len() > MAX_ASR_EMISSIONS_PER_REQUEST {
        return Err(format!(
            "native ASR response exceeds {MAX_ASR_EMISSIONS_PER_REQUEST} emissions"
        ));
    }

    let mut revisions = BTreeMap::<String, (u32, bool)>::new();
    let mut has_final = false;
    for emission in &response.emissions {
        emission.validate().map_err(|error| {
            format!("native ASR response contains an invalid emission: {error}")
        })?;
        if emission.capture_start_ns < job.started_at.monotonic_ns
            || emission.capture_end_ns > job.ended_at.monotonic_ns
        {
            return Err("native ASR emission falls outside its job capture range".to_owned());
        }

        let history = revisions
            .entry(emission.utterance_key.clone())
            .or_insert((0, false));
        if history.1 {
            return Err("a native ASR utterance cannot emit after its final revision".to_owned());
        }
        if emission.revision <= history.0 {
            return Err("native ASR emission revisions must increase".to_owned());
        }
        history.0 = emission.revision;
        history.1 = emission.kind == TranscriptEmissionKind::Final;
        has_final |= history.1;
    }
    Ok(has_final)
}

fn validate_native_inference_gap(
    runtime: &DispatcherRuntime,
    gap: &InferenceGap,
) -> Result<(), String> {
    gap.validate()
        .map_err(|error| format!("native inference gap is invalid: {error}"))?;
    if gap.session_id != runtime.session_id {
        return Err(
            "native inference gap session does not match its dispatcher runtime".to_owned(),
        );
    }
    if gap.runtime_id != runtime.id.as_uuid() {
        return Err(
            "native inference gap generation does not match its dispatcher runtime".to_owned(),
        );
    }
    if gap.capture_segment_id != runtime.capture_segment_id {
        return Err(
            "native inference gap segment does not match its dispatcher runtime".to_owned(),
        );
    }
    validate_native_capture_range(
        runtime,
        &gap.started_at,
        &gap.ended_at,
        "native inference gap",
    )
}

fn validate_native_capture_range(
    runtime: &DispatcherRuntime,
    started_at: &CapturePoint,
    ended_at: &CapturePoint,
    label: &str,
) -> Result<(), String> {
    if ended_at.monotonic_ns < started_at.monotonic_ns {
        return Err(format!("{label} capture range is inverted"));
    }
    if ended_at.wall_clock < started_at.wall_clock {
        return Err(format!("{label} wall-clock range is inverted"));
    }

    for (endpoint, point) in [("start", started_at), ("end", ended_at)] {
        let expected = native_capture_point_at(runtime, point.monotonic_ns)?;
        if point.wall_clock != expected.wall_clock {
            return Err(format!(
                "{label} {endpoint} wall-clock does not match its capture segment clock"
            ));
        }
    }
    Ok(())
}

fn native_capture_point_at(
    runtime: &DispatcherRuntime,
    monotonic_ns: u64,
) -> Result<CapturePoint, String> {
    let offset_ns = monotonic_ns
        .checked_sub(runtime.capture_anchor.monotonic_ns)
        .ok_or_else(|| "native capture time precedes its capture segment".to_owned())?;
    let offset_ns = i64::try_from(offset_ns)
        .map_err(|_| "native capture time exceeds the supported wall-clock range".to_owned())?;
    Ok(CapturePoint {
        monotonic_ns,
        wall_clock: runtime.capture_anchor.wall_clock + chrono::Duration::nanoseconds(offset_ns),
    })
}

fn timeline_projections(revisions: Vec<TranscriptRevision>) -> BTreeMap<Uuid, Vec<TranscriptSpan>> {
    let mut latest_by_span = BTreeMap::<(Uuid, Uuid), TranscriptSpan>::new();
    for revision in revisions.into_iter().filter(|revision| revision.is_final) {
        let projection = transcript_revision_projection(&revision);
        let key = (projection.session_id, projection.id);
        match latest_by_span.get(&key) {
            Some(previous) if previous.revision > projection.revision => {}
            _ => {
                latest_by_span.insert(key, projection);
            }
        }
    }

    let mut timelines = BTreeMap::<Uuid, Vec<TranscriptSpan>>::new();
    for ((session_id, _), projection) in latest_by_span {
        timelines.entry(session_id).or_default().push(projection);
    }
    for timeline in timelines.values_mut() {
        timeline.sort_by_key(|span| (span.capture_start_ns, span.revision));
    }
    timelines
}

/// Returns the current durable final revision for exactly one logical span.
/// Revisions are append-only and SQLite validates strict consecutive revision
/// numbers, so the largest revision is the only current head.
fn durable_current_final_transcript_head(
    revisions: &[TranscriptRevision],
    logical_span_id: Uuid,
) -> Option<&TranscriptRevision> {
    revisions
        .iter()
        .filter(|revision| revision.logical_span_id == logical_span_id && revision.is_final)
        .max_by_key(|revision| revision.revision)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::audio::{CaptureGap, CaptureGapReason};
    use crate::audio::{CapturePoint, DispatcherRuntimeId};
    #[cfg(target_os = "macos")]
    use crate::domain::CaptureSegment;
    use crate::domain::{
        SpeakerClusterAliasRevision, TranscriptModelProvenance, TranscriptRevision,
        TranscriptSource, TranscriptTiming,
    };
    use crate::inference::model_registry::{
        LicenseAcknowledgement, LocalModelKind, ModelImportRequest,
    };
    use crate::inference::{
        AsrEngine, AsrRequest, FixtureAsr, InferenceAudioWindow, InferenceGap, InferenceGapReason,
        InferenceGapStage, ModelProvenance, TranscriptEmissionKind, TranscriptEmissionMapper,
        INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ, WHISPER_CPP_GGML_INPUT_FORMAT,
    };
    use chrono::Duration;
    use sha2::{Digest, Sha256};

    #[test]
    fn starts_and_stops_an_audited_local_session() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();

        assert_eq!(
            state.privacy_status().unwrap().recording_session_id,
            Some(session.id)
        );
        assert!(state.stop_session().unwrap().is_some());
        assert!(state.audit_is_valid().unwrap());
    }

    fn native_runtime_for(session: &CaptureSession, capture_segment_id: Uuid) -> DispatcherRuntime {
        let anchor_offset_ns = 1_000_000;
        DispatcherRuntime::new(
            DispatcherRuntimeId::generate(),
            session.id,
            capture_segment_id,
            CapturePoint {
                monotonic_ns: session.started_monotonic_ns + anchor_offset_ns,
                wall_clock: session.started_at + Duration::nanoseconds(anchor_offset_ns as i64),
            },
        )
        .unwrap()
    }

    fn native_point(runtime: &DispatcherRuntime, offset_ns: u64) -> CapturePoint {
        CapturePoint {
            monotonic_ns: runtime.capture_anchor.monotonic_ns + offset_ns,
            wall_clock: runtime.capture_anchor.wall_clock + Duration::nanoseconds(offset_ns as i64),
        }
    }

    fn install_starting_native_session(state: &AppState) -> (CaptureSession, DispatcherRuntime) {
        let session = CaptureSession::begin_starting_with_id(
            Uuid::new_v4(),
            1_000,
            DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(1),
        )
        .unwrap();
        state
            .sessions
            .lock()
            .unwrap()
            .insert(session.id, session.clone());
        state
            .timelines
            .lock()
            .unwrap()
            .entry(session.id)
            .or_default();

        let runtime = native_runtime_for(&session, Uuid::new_v4());
        *state.native_runtime.lock().unwrap() = Some(NativeRuntimeFence {
            runtime: runtime.clone(),
            phase: NativeRuntimePhase::Active,
            capture_input_stopped_audited: false,
            capture_stop_point: None,
        });
        (session, runtime)
    }

    #[test]
    fn native_starting_session_is_private_until_dispatcher_arm_publishes_it() {
        let state = AppState::in_memory();
        let (session, runtime) = install_starting_native_session(&state);

        assert_eq!(session.state, crate::domain::SessionState::Starting);
        assert_eq!(state.privacy_status().unwrap().recording_session_id, None);
        assert!(state.active_recording_session().unwrap().is_none());
        assert_eq!(
            state.active_live_session().unwrap().map(|live| live.id),
            Some(session.id)
        );

        let published = state.publish_native_capture_recording(&runtime).unwrap();
        assert_eq!(published.state, crate::domain::SessionState::Recording);
        assert_eq!(
            state.privacy_status().unwrap().recording_session_id,
            Some(session.id)
        );
    }

    #[test]
    fn native_outcome_can_finish_while_a_started_session_is_still_starting() {
        let state = AppState::in_memory();
        let (session, runtime) = install_starting_native_session(&state);
        let job = native_job(&runtime, Uuid::new_v4());
        let response = AsrResponse {
            emissions: vec![native_emission(&runtime, TranscriptEmissionKind::Final, 1)],
        };

        let projections = state
            .persist_native_outcome(&runtime, &AsrOutcome::Response { job, response })
            .unwrap();

        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0].session_id, session.id);
    }

    #[test]
    fn stopping_a_starting_session_does_not_leave_it_live() {
        let state = AppState::in_memory();
        let (session, _) = install_starting_native_session(&state);

        let stopped = state
            .stop_session()
            .unwrap()
            .expect("starting session stops");

        assert_eq!(stopped.id, session.id);
        assert_eq!(stopped.state, crate::domain::SessionState::Stopped);
        assert_eq!(state.privacy_status().unwrap().recording_session_id, None);
    }

    #[test]
    fn native_stop_events_reuse_one_capture_clock_point_until_session_stop_commits() {
        let state = AppState::in_memory();
        let (session, runtime) = install_starting_native_session(&state);
        state.begin_native_runtime_shutdown(&runtime).unwrap();

        let first = state
            .capture_native_stop_point(&runtime, runtime.capture_anchor.monotonic_ns + 2_000_000)
            .unwrap();
        let retry = state
            .capture_native_stop_point(&runtime, runtime.capture_anchor.monotonic_ns + 9_000_000)
            .unwrap();
        assert_eq!(retry, first);

        state.begin_native_runtime_handoff(&runtime).unwrap();
        state.mark_native_runtime_drained(&runtime).unwrap();
        assert_eq!(
            state.finish_drained_native_runtime(&runtime).unwrap(),
            first
        );

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate session stop audit persistence failure after native drain");
        }))
        .is_err());
        assert!(state.finish_capture_session_at(first.clone()).is_err());
        {
            let fence = state.native_runtime.lock().unwrap();
            let fence = fence.as_ref().expect("drained fence remains retryable");
            assert_eq!(fence.phase, NativeRuntimePhase::Drained);
            assert_eq!(fence.capture_stop_point, Some(first.clone()));
            assert!(fence.capture_input_stopped_audited);
        }

        state.audit_store.clear_poison();
        assert_eq!(
            state.finish_drained_native_runtime(&runtime).unwrap(),
            first
        );
        let stopped = state
            .finish_capture_session_at(first.clone())
            .unwrap()
            .expect("starting session stops after native drain");
        assert_eq!(stopped.id, session.id);
        assert_eq!(stopped.stopped_at, Some(first.wall_clock));

        let events = state.audit_trail.lock().unwrap().events().to_vec();
        for kind in [AuditKind::CaptureInputStopped, AuditKind::SessionStopped] {
            let matching = events
                .iter()
                .filter(|event| event.kind == kind)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), 1, "{kind:?} is only recorded once");
            let event = matching[0];
            assert_eq!(event.monotonic_ns, first.monotonic_ns);
            assert_eq!(event.wall_clock, first.wall_clock);
        }

        state.clear_native_runtime_after_drain(&runtime).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn capture_projection_retries_a_drained_stop_after_session_audit_failure() {
        let state = AppState::in_memory();
        let (session, runtime) = install_starting_native_session(&state);
        state.begin_native_runtime_shutdown(&runtime).unwrap();
        let stop_point = state
            .capture_native_stop_point(&runtime, runtime.capture_anchor.monotonic_ns + 2_000_000)
            .unwrap();
        state.begin_native_runtime_handoff(&runtime).unwrap();
        state.mark_native_runtime_drained(&runtime).unwrap();

        // This is the successful portion of the first stop attempt. The
        // following session audit is the part that transiently fails.
        assert_eq!(
            state.finish_drained_native_runtime(&runtime).unwrap(),
            stop_point
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate SessionStopped persistence failure after native drain");
        }))
        .is_err());

        let error = state.capture_projection().unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert_eq!(
            state.active_live_session().unwrap().map(|live| live.id),
            Some(session.id)
        );
        assert_eq!(
            state
                .native_runtime
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .capture_stop_point,
            Some(stop_point.clone())
        );

        state.audit_store.clear_poison();
        let projection = state.capture_projection().unwrap();
        assert_eq!(projection.status, crate::audio::CaptureStatus::Idle);
        assert!(state.active_live_session().unwrap().is_none());
        assert!(state.native_runtime.lock().unwrap().is_none());

        let events = state.audit_trail.lock().unwrap().events().to_vec();
        for kind in [AuditKind::CaptureInputStopped, AuditKind::SessionStopped] {
            let matching = events
                .iter()
                .filter(|event| event.kind == kind)
                .collect::<Vec<_>>();
            assert_eq!(
                matching.len(),
                1,
                "{kind:?} is recorded once across projection retry"
            );
            assert_eq!(matching[0].monotonic_ns, stop_point.monotonic_ns);
            assert_eq!(matching[0].wall_clock, stop_point.wall_clock);
        }
    }

    #[test]
    fn session_stop_keeps_the_session_live_when_audit_persistence_fails() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let point = CapturePoint {
            monotonic_ns: session.started_monotonic_ns + 1_000,
            wall_clock: session.started_at + Duration::microseconds(1),
        };

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate session stop audit persistence failure");
        }))
        .is_err());
        let error = state.finish_capture_session_at(point.clone()).unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert_eq!(
            state.privacy_status().unwrap().recording_session_id,
            Some(session.id)
        );
        assert_eq!(
            state
                .sessions
                .lock()
                .unwrap()
                .get(&session.id)
                .unwrap()
                .state,
            crate::domain::SessionState::Recording
        );

        state.audit_store.clear_poison();
        let stopped = state
            .finish_capture_session_at(point)
            .unwrap()
            .expect("retry stops the live session");
        assert_eq!(stopped.state, crate::domain::SessionState::Stopped);
        assert_eq!(state.privacy_status().unwrap().recording_session_id, None);
    }

    fn native_job(runtime: &DispatcherRuntime, id: Uuid) -> AsrJobMetadata {
        AsrJobMetadata {
            id,
            session_id: runtime.session_id,
            runtime_id: runtime.id,
            capture_segment_id: runtime.capture_segment_id,
            started_at: native_point(runtime, 100_000),
            ended_at: native_point(runtime, 900_000),
        }
    }

    fn native_emission(
        runtime: &DispatcherRuntime,
        kind: TranscriptEmissionKind,
        revision: u32,
    ) -> TranscriptEmission {
        TranscriptEmission {
            utterance_key: "native-window-1".to_owned(),
            capture_start_ns: runtime.capture_anchor.monotonic_ns + 200_000,
            capture_end_ns: runtime.capture_anchor.monotonic_ns + 800_000,
            text: "本地结果。".to_owned(),
            kind,
            revision,
            word_timings: Vec::new(),
            model_provenance: ModelProvenance::new(
                "word-covenant",
                "fixture-local-asr",
                "1",
                "a".repeat(64),
            )
            .unwrap(),
        }
    }

    #[test]
    fn persists_a_fenced_native_final_using_its_capture_segment_anchor() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let runtime = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(runtime.clone()).unwrap();

        let job = native_job(&runtime, Uuid::new_v4());
        let response = AsrResponse {
            emissions: vec![native_emission(&runtime, TranscriptEmissionKind::Final, 1)],
        };
        let projections = state
            .persist_native_outcome(&runtime, &AsrOutcome::Response { job, response })
            .unwrap();

        assert_eq!(projections.len(), 1);
        assert_eq!(
            projections[0].wall_clock_start,
            Some(runtime.capture_anchor.wall_clock + Duration::nanoseconds(200_000))
        );
        assert_eq!(state.list_timeline(Some(session.id)).unwrap(), projections);
        let final_projection = state
            .final_transcript_projection()
            .expect("the native final publishes a timeline refresh reference");
        assert_eq!(
            final_projection,
            FinalTranscriptProjection {
                session_id: session.id,
                revision: 1,
            }
        );
        assert_eq!(
            serde_json::to_value(final_projection).unwrap(),
            serde_json::json!({ "sessionId": session.id.to_string(), "revision": 1 })
        );
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn turns_an_asr_response_without_a_final_into_a_durable_gap() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let runtime = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(runtime.clone()).unwrap();

        let job = native_job(&runtime, Uuid::new_v4());
        let response = AsrResponse {
            emissions: vec![native_emission(
                &runtime,
                TranscriptEmissionKind::Partial,
                1,
            )],
        };
        let projections = state
            .persist_native_outcome(
                &runtime,
                &AsrOutcome::Response {
                    job: job.clone(),
                    response,
                },
            )
            .unwrap();

        assert!(projections.is_empty());
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());
        assert_eq!(state.final_transcript_projection(), None);
        let store = state.audit_store.lock().unwrap();
        let gaps = store.list_inference_gaps(session.id).unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].job_id, Some(job.id));
        assert_eq!(gaps[0].stage, InferenceGapStage::Worker);
        assert_eq!(gaps[0].reason, InferenceGapReason::EngineFailed);
        drop(store);

        let final_projections = state
            .persist_native_outcome(
                &runtime,
                &AsrOutcome::Response {
                    job: native_job(&runtime, Uuid::new_v4()),
                    response: AsrResponse {
                        emissions: vec![native_emission(
                            &runtime,
                            TranscriptEmissionKind::Final,
                            1,
                        )],
                    },
                },
            )
            .unwrap();
        assert_eq!(final_projections.len(), 1);
    }

    #[test]
    fn turns_a_malformed_native_response_into_a_durable_gap() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let runtime = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(runtime.clone()).unwrap();

        let job = native_job(&runtime, Uuid::new_v4());
        let response = AsrResponse {
            emissions: vec![
                native_emission(&runtime, TranscriptEmissionKind::Final, 1),
                native_emission(&runtime, TranscriptEmissionKind::Partial, 2),
            ],
        };
        let projections = state
            .persist_native_outcome(
                &runtime,
                &AsrOutcome::Response {
                    job: job.clone(),
                    response,
                },
            )
            .unwrap();

        assert!(projections.is_empty());
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());
        let gaps = state
            .audit_store
            .lock()
            .unwrap()
            .list_inference_gaps(session.id)
            .unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].job_id, Some(job.id));
        assert_eq!(gaps[0].stage, InferenceGapStage::Worker);
        assert_eq!(gaps[0].reason, InferenceGapReason::EngineFailed);
    }

    #[test]
    fn rejects_a_native_outcome_from_a_cleared_generation() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let original = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(original.clone()).unwrap();
        state.begin_native_runtime_shutdown(&original).unwrap();
        state.begin_native_runtime_handoff(&original).unwrap();
        state.mark_native_runtime_drained(&original).unwrap();
        state.clear_native_runtime_after_drain(&original).unwrap();

        let replacement = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(replacement).unwrap();
        let stale_gap = InferenceGap::new(
            Uuid::new_v4(),
            original.session_id,
            original.id.as_uuid(),
            original.capture_segment_id,
            None,
            native_point(&original, 100_000),
            native_point(&original, 100_000),
            InferenceGapStage::Shutdown,
            InferenceGapReason::StoppedBeforeInference,
        )
        .unwrap();

        let error = state
            .persist_native_outcome(&original, &AsrOutcome::Gap(stale_gap))
            .unwrap_err();
        assert!(error.contains("does not match the active fence"));
        assert!(state
            .audit_store
            .lock()
            .unwrap()
            .list()
            .unwrap()
            .iter()
            .all(|event| { event.kind != AuditKind::InferenceGapRecorded }));
    }

    #[test]
    fn rejects_a_native_outcome_once_runtime_handoff_begins() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let runtime = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(runtime.clone()).unwrap();
        state.begin_native_runtime_shutdown(&runtime).unwrap();
        state.begin_native_runtime_handoff(&runtime).unwrap();

        let gap = InferenceGap::new(
            Uuid::new_v4(),
            runtime.session_id,
            runtime.id.as_uuid(),
            runtime.capture_segment_id,
            None,
            native_point(&runtime, 100_000),
            native_point(&runtime, 100_000),
            InferenceGapStage::Shutdown,
            InferenceGapReason::StoppedBeforeInference,
        )
        .unwrap();
        let error = state
            .persist_native_outcome(&runtime, &AsrOutcome::Gap(gap))
            .unwrap_err();
        assert!(error.contains("after runtime drain began"));
        assert!(state
            .audit_store
            .lock()
            .unwrap()
            .list()
            .unwrap()
            .iter()
            .all(|event| event.kind != AuditKind::InferenceGapRecorded));
    }

    #[test]
    fn native_final_can_retry_after_a_transient_persistence_failure() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let runtime = native_runtime_for(&session, Uuid::new_v4());
        state.activate_native_runtime(runtime.clone()).unwrap();

        let outcome = AsrOutcome::Response {
            job: native_job(&runtime, Uuid::new_v4()),
            response: AsrResponse {
                emissions: vec![native_emission(&runtime, TranscriptEmissionKind::Final, 1)],
            },
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate a transient native outcome persistence failure");
        }))
        .is_err());
        let error = state
            .persist_native_outcome(&runtime, &outcome)
            .unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());
        assert_eq!(state.final_transcript_projection(), None);

        state.audit_store.clear_poison();
        let persisted = state.persist_native_outcome(&runtime, &outcome).unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(
            state.final_transcript_projection(),
            Some(FinalTranscriptProjection {
                session_id: session.id,
                revision: 1,
            })
        );
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn persists_reopens_and_audits_inference_gaps_through_application_state() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-state-inference-gap-{}.sqlite3",
            Uuid::new_v4()
        ));
        let gap = {
            let state = AppState::open(&database).unwrap();
            let session = state.start_session().unwrap();
            let started_at = CapturePoint {
                monotonic_ns: 5_000,
                wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(5),
            };
            let gap = InferenceGap::new(
                Uuid::new_v4(),
                session.id,
                Uuid::new_v4(),
                Uuid::new_v4(),
                Some(Uuid::new_v4()),
                started_at.clone(),
                CapturePoint {
                    monotonic_ns: 9_000,
                    wall_clock: started_at.wall_clock + Duration::milliseconds(4),
                },
                InferenceGapStage::JobQueue,
                InferenceGapReason::JobQueueSaturated,
            )
            .unwrap();

            state.record_inference_gap(&gap).unwrap();
            // A retry after the dispatcher lost its lease acknowledgement
            // must reuse the durable binding rather than extend the chain.
            state.record_inference_gap(&gap).unwrap();

            {
                let store = state.audit_store.lock().unwrap();
                assert_eq!(
                    store.list_inference_gaps(session.id).unwrap(),
                    vec![gap.clone()]
                );
                assert!(store.verify().unwrap());
            }
            let event = {
                let trail = state.audit_trail.lock().unwrap();
                trail
                    .events()
                    .iter()
                    .find(|event| event.kind == AuditKind::InferenceGapRecorded)
                    .cloned()
                    .expect("the inference gap is audit recorded")
            };
            assert_eq!(event.run_id, Some(session.id));
            assert_eq!(event.causation_id, gap.job_id);
            assert_eq!(event.monotonic_ns, gap.ended_at.monotonic_ns);
            assert_eq!(event.wall_clock, gap.ended_at.wall_clock);
            assert!(event.matches_payload(&gap).unwrap());
            assert!(state.audit_is_valid().unwrap());

            // Model a process-local interruption after SQLite committed the
            // transaction but before the matching event was appended to the
            // in-memory trail. A replay restores that exact durable event.
            let events_without_gap = {
                let trail = state.audit_trail.lock().unwrap();
                trail
                    .events()
                    .iter()
                    .filter(|candidate| candidate.id != event.id)
                    .cloned()
                    .collect()
            };
            *state.audit_trail.lock().unwrap() = AuditTrail::from_events(events_without_gap);
            state.record_inference_gap(&gap).unwrap();
            assert!(state.audit_is_valid().unwrap());

            let mut conflicting = gap.clone();
            conflicting.reason = InferenceGapReason::EngineFailed;
            let error = state.record_inference_gap(&conflicting).unwrap_err();
            assert!(error.contains("different immutable payload"));
            assert_eq!(
                state
                    .audit_store
                    .lock()
                    .unwrap()
                    .list()
                    .unwrap()
                    .iter()
                    .filter(|candidate| candidate.kind == AuditKind::InferenceGapRecorded)
                    .count(),
                1
            );

            gap
        };

        let reopened = AppState::open(&database).unwrap();
        reopened.record_inference_gap(&gap).unwrap();
        {
            let store = reopened.audit_store.lock().unwrap();
            assert_eq!(
                store.list_inference_gaps(gap.session_id).unwrap(),
                vec![gap.clone()]
            );
            assert_eq!(
                store
                    .list()
                    .unwrap()
                    .iter()
                    .filter(|candidate| candidate.kind == AuditKind::InferenceGapRecorded)
                    .count(),
                1
            );
            assert!(store.verify().unwrap());
        }
        assert!(reopened.audit_is_valid().unwrap());
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persists_capture_segment_and_gap_through_the_application_state() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let anchor = CapturePoint {
            monotonic_ns: 5_000,
            wall_clock: Utc::now(),
        };
        let segment = CaptureSegment::new(
            session.id,
            "built-in-mic",
            "Built-in Microphone",
            48_000,
            2,
            anchor.monotonic_ns,
            anchor.wall_clock,
        )
        .unwrap();
        let gap = CaptureGap {
            started_at: CapturePoint {
                monotonic_ns: 8_000,
                wall_clock: anchor.wall_clock + Duration::milliseconds(3),
            },
            ended_at: CapturePoint {
                monotonic_ns: 9_500,
                wall_clock: anchor.wall_clock + Duration::milliseconds(4),
            },
            reason: CaptureGapReason::InputDeviceUnavailable,
        };

        state.record_capture_segment(&segment).unwrap();
        state.record_capture_gap(session.id, &gap).unwrap();

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_capture_segments(session.id).unwrap(),
            vec![segment]
        );
        assert_eq!(store.list_capture_gaps(session.id).unwrap(), vec![gap]);
        assert!(store.verify().unwrap());
        drop(store);
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn development_mock_persists_only_final_local_asr_results_and_stops_cleanly() {
        let state = AppState::in_memory();
        let before = state.privacy_status().unwrap();
        let session = state.start_development_mock_session().unwrap();

        // The first 2.8-second scripted sentence has ended after 140 packets,
        // but its partial text remains inside the native inference mapper
        // until the pipeline's silence hangover finalizes it.
        for _ in 0..14 {
            let progress = state.advance_development_mock(10).unwrap();
            assert_eq!(progress.session_id, session.id);
            assert!(progress.spans.is_empty());
        }
        {
            let store = state.audit_store.lock().unwrap();
            assert!(store
                .list_transcript_revisions(session.id)
                .unwrap()
                .is_empty());
            assert!(store
                .search_transcript_revisions(Some(session.id), "partialonlyone")
                .unwrap()
                .is_empty());
        }

        let first_final = state.advance_development_mock(10).unwrap();
        assert_eq!(first_final.spans.len(), 1);
        assert_eq!(first_final.spans[0].session_id, session.id);
        assert_eq!(
            first_final.spans[0].capture_start_ns,
            session.started_monotonic_ns
        );
        assert_eq!(
            first_final.spans[0].capture_end_ns - first_final.spans[0].capture_start_ns,
            2_800_000_000
        );
        assert_eq!(
            first_final.spans[0].source,
            TranscriptSource::LocalInference
        );
        assert_eq!(first_final.spans[0].text, "本次记录仅保存在本机。");
        assert_eq!(first_final.spans[0].speaker_cluster_id, None);

        let mut spans = first_final.spans;
        loop {
            let progress = state.advance_development_mock(10).unwrap();
            assert_eq!(progress.session_id, session.id);
            assert!(progress.packets_advanced <= 10);
            spans.extend(progress.spans);
            if progress.exhausted {
                break;
            }
        }

        assert_eq!(spans.len(), 3);
        assert!(spans
            .iter()
            .all(|span| span.session_id == session.id
                && span.source == TranscriptSource::LocalInference));
        assert_eq!(spans[0].capture_start_ns, session.started_monotonic_ns);
        assert_eq!(
            spans[0].capture_end_ns - spans[0].capture_start_ns,
            2_800_000_000
        );
        assert_eq!(spans[2].text, "先生成一份待确认的行动草案。");
        assert_eq!(state.list_timeline(Some(session.id)).unwrap().len(), 3);
        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            3
        );
        assert!(store
            .search_transcript_revisions(Some(session.id), "partialonlyone")
            .unwrap()
            .is_empty());
        assert!(store.verify().unwrap());
        drop(store);
        assert!(state.audit_is_valid().unwrap());

        assert!(state.stop_session().unwrap().is_some());
        assert!(state.advance_development_mock(1).is_err());

        let after = state.privacy_status().unwrap();
        assert_eq!(after.local_only, before.local_only);
        assert_eq!(after.egress_enabled, before.egress_enabled);
        assert_eq!(
            after.active_egress_approvals,
            before.active_egress_approvals
        );
    }

    #[test]
    fn development_mock_retries_a_pending_response_after_persistence_failure() {
        let state = AppState::in_memory();
        let session = state.start_development_mock_session().unwrap();
        for _ in 0..14 {
            assert!(state.advance_development_mock(10).unwrap().spans.is_empty());
        }

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate a transient mock transcript persistence failure");
        }))
        .is_err());
        let error = state.advance_development_mock(10).unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());

        state.audit_store.clear_poison();
        let retry = state.advance_development_mock(1).unwrap();
        assert_eq!(retry.spans.len(), 1);
        assert_eq!(retry.spans[0].text, "本次记录仅保存在本机。");

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            1
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn development_mock_flushes_before_session_stopped_and_retries_stop_failures() {
        let state = AppState::in_memory();
        let session = state.start_development_mock_session().unwrap();
        for _ in 0..14 {
            assert!(state.advance_development_mock(10).unwrap().spans.is_empty());
        }

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate a transient stop-flush persistence failure");
        }))
        .is_err());
        let error = state.stop_session().unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert_eq!(
            state.privacy_status().unwrap().recording_session_id,
            Some(session.id)
        );
        assert!(!state
            .audit_trail
            .lock()
            .unwrap()
            .events()
            .iter()
            .any(|event| event.kind == AuditKind::SessionStopped));

        state.audit_store.clear_poison();
        assert_eq!(state.stop_session().unwrap().unwrap().id, session.id);

        let events = state.audit_trail.lock().unwrap().events().to_vec();
        let transcript_index = events
            .iter()
            .position(|event| event.kind == AuditKind::TranscriptRevisionRecorded)
            .expect("the mock final was persisted before stopping");
        let stopped_index = events
            .iter()
            .position(|event| event.kind == AuditKind::SessionStopped)
            .expect("the session stop event was recorded");
        assert!(transcript_index < stopped_index);

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            1
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn replaying_a_scripted_mock_final_does_not_create_a_second_revision() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let mut runner = DevelopmentMockRunner::new(&session).unwrap();
        let response = (0..15)
            .find_map(|_| {
                runner.advance(10).unwrap();
                runner
                    .begin_pending_asr_delivery()
                    .unwrap()
                    .map(|delivery| delivery.response)
            })
            .expect("the first scripted sentence finalizes within 15 advances");

        assert_eq!(
            state
                .append_local_asr_response(session.id, response.clone())
                .unwrap()
                .len(),
            1
        );
        let event_count = state.audit_trail.lock().unwrap().events().len();
        assert!(state
            .append_local_asr_response(session.id, response)
            .unwrap()
            .is_empty());
        assert_eq!(
            state.audit_trail.lock().unwrap().events().len(),
            event_count
        );

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            1
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn durable_asr_idempotency_survives_mapper_loss_and_rejects_payload_conflicts() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let mut runner = DevelopmentMockRunner::new(&session).unwrap();
        let response = (0..15)
            .find_map(|_| {
                runner.advance(10).unwrap();
                runner
                    .begin_pending_asr_delivery()
                    .unwrap()
                    .map(|delivery| delivery.response)
            })
            .expect("the first scripted sentence finalizes within 15 advances");

        assert_eq!(
            state
                .append_local_asr_response(session.id, response.clone())
                .unwrap()
                .len(),
            1
        );

        // Model a process crash after SQLite commits but before the
        // in-memory mapper could retain any completion history.
        *state.inference_mapper.lock().unwrap() = TranscriptEmissionMapper::default();
        assert!(state
            .append_local_asr_response(session.id, response.clone())
            .unwrap()
            .is_empty());

        let event_count = state.audit_trail.lock().unwrap().events().len();
        let mut conflicting = response;
        let final_emission = conflicting
            .emissions
            .iter_mut()
            .find(|emission| emission.kind == TranscriptEmissionKind::Final)
            .expect("scripted response includes a final emission");
        final_emission.text = "冲突的重放内容。".to_owned();

        *state.inference_mapper.lock().unwrap() = TranscriptEmissionMapper::default();
        let error = state
            .append_local_asr_response(session.id, conflicting)
            .unwrap_err();
        assert!(error.contains("conflicts with its durable idempotency record"));
        assert_eq!(
            state.audit_trail.lock().unwrap().events().len(),
            event_count
        );

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            1
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn persists_a_final_fixture_asr_output_as_an_audited_revision() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let capture_start_ns = session.started_monotonic_ns;
        let capture_end_ns = capture_start_ns + 1_000_000_000;
        let window = InferenceAudioWindow::new(
            session.id,
            capture_start_ns,
            capture_end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; INFERENCE_SAMPLE_RATE_HZ as usize],
        )
        .unwrap();
        let request = AsrRequest::new(window, Some("zh".to_owned()), true).unwrap();
        let mut fixture = FixtureAsr::default();
        let output = fixture.transcribe(&request).unwrap();
        let projections = state.append_local_asr_response(session.id, output).unwrap();

        let timeline = state.list_timeline(Some(session.id)).unwrap();
        assert_eq!(timeline.len(), 1);
        assert_eq!(projections, timeline);
        assert_eq!(timeline[0].text, "本次记录仅保存在本机。");
        assert_eq!(timeline[0].source, TranscriptSource::LocalInference);

        let store = state.audit_store.lock().unwrap();
        let revisions = store.list_transcript_revisions(session.id).unwrap();
        assert_eq!(revisions.len(), 1);
        assert_eq!(revisions[0].logical_span_id, timeline[0].id);
        assert_eq!(revisions[0].revision, 1);
        assert_eq!(revisions[0].text, "本次记录仅保存在本机。");
        assert!(store.verify().unwrap());
        drop(store);
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn audit_status_detects_a_tampered_durable_asr_binding() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-audit-status-binding-{}.sqlite3",
            Uuid::new_v4()
        ));
        let state = AppState::open(&database).unwrap();
        let session = state.start_session().unwrap();
        let capture_start_ns = session.started_monotonic_ns;
        let capture_end_ns = capture_start_ns + 1_000_000_000;
        let window = InferenceAudioWindow::new(
            session.id,
            capture_start_ns,
            capture_end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; INFERENCE_SAMPLE_RATE_HZ as usize],
        )
        .unwrap();
        let request = AsrRequest::new(window, Some("zh".to_owned()), true).unwrap();
        let mut fixture = FixtureAsr::default();
        let output = fixture.transcribe(&request).unwrap();
        state.append_local_asr_response(session.id, output).unwrap();
        assert!(state.audit_is_valid().unwrap());

        let revision = state
            .audit_store
            .lock()
            .unwrap()
            .list_transcript_revisions(session.id)
            .unwrap()
            .into_iter()
            .next()
            .expect("fixture final is durable");
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute_batch("DROP TRIGGER asr_final_idempotency_is_immutable_delete;")
            .unwrap();
        connection
            .execute(
                "DELETE FROM asr_final_idempotency WHERE revision_id = ?1",
                rusqlite::params![revision.id.to_string()],
            )
            .unwrap();
        drop(connection);

        assert!(!state.audit_is_valid().unwrap());
        drop(state);
        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn retries_a_final_asr_response_after_persistence_failure_without_duplicates() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let capture_start_ns = session.started_monotonic_ns;
        let capture_end_ns = capture_start_ns + 1_000_000_000;
        let window = InferenceAudioWindow::new(
            session.id,
            capture_start_ns,
            capture_end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; INFERENCE_SAMPLE_RATE_HZ as usize],
        )
        .unwrap();
        let request = AsrRequest::new(window, Some("zh".to_owned()), true).unwrap();
        let mut fixture = FixtureAsr::default();
        let response = fixture.transcribe(&request).unwrap();
        let final_response = AsrResponse {
            emissions: vec![response
                .emissions
                .last()
                .cloned()
                .expect("fixture emits one final result")],
        };

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _store = state.audit_store.lock().expect("audit store is available");
            panic!("simulate a transient transcript persistence failure");
        }))
        .is_err());
        let error = state
            .append_local_asr_response(session.id, response)
            .unwrap_err();
        assert!(error.contains("audit store lock poisoned"));
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());

        state.audit_store.clear_poison();
        let persisted = state
            .append_local_asr_response(session.id, final_response.clone())
            .unwrap();
        assert_eq!(persisted.len(), 1);
        assert!(state
            .append_local_asr_response(session.id, final_response)
            .unwrap()
            .is_empty());

        let store = state.audit_store.lock().unwrap();
        assert_eq!(
            store.list_transcript_revisions(session.id).unwrap().len(),
            1
        );
        assert!(store.verify().unwrap());
        drop(store);
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn retries_a_final_asr_response_when_the_timeline_lock_fails_before_commit() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let capture_start_ns = session.started_monotonic_ns;
        let capture_end_ns = capture_start_ns + 1_000_000_000;
        let window = InferenceAudioWindow::new(
            session.id,
            capture_start_ns,
            capture_end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; INFERENCE_SAMPLE_RATE_HZ as usize],
        )
        .unwrap();
        let request = AsrRequest::new(window, Some("zh".to_owned()), false).unwrap();
        let mut fixture = FixtureAsr::default();
        let response = fixture.transcribe(&request).unwrap();

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _timelines = state.timelines.lock().expect("timeline state is available");
            panic!("simulate a transient timeline lock failure");
        }))
        .is_err());
        let error = state
            .append_local_asr_response(session.id, response.clone())
            .unwrap_err();
        assert!(error.contains("timeline state lock poisoned"));

        state.timelines.clear_poison();
        let store = state.audit_store.lock().unwrap();
        assert!(store
            .list_transcript_revisions(session.id)
            .unwrap()
            .is_empty());
        drop(store);

        let persisted = state
            .append_local_asr_response(session.id, response)
            .unwrap();
        assert_eq!(persisted.len(), 1);
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn does_not_persist_partial_inference_results() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let revision = TranscriptRevision::original(
            session.id,
            TranscriptTiming::new(
                session.started_monotonic_ns,
                session.started_monotonic_ns + 1,
                session.started_at,
                session.started_at,
            )
            .unwrap(),
            None,
            "临时输出",
            false,
            TranscriptSource::Synthetic,
            None,
            None,
        )
        .unwrap();

        assert!(state.append_final_transcript_revision(revision).is_err());
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn rejects_direct_local_inference_revisions_without_an_asr_binding() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let revision = TranscriptRevision::original(
            session.id,
            TranscriptTiming::new(
                session.started_monotonic_ns,
                session.started_monotonic_ns + 1_000_000_000,
                session.started_at,
                session.started_at + Duration::seconds(1),
            )
            .unwrap(),
            None,
            "未经映射的本地推理结果",
            true,
            TranscriptSource::LocalInference,
            Some(
                TranscriptModelProvenance::new(
                    "fixture",
                    "fixture-asr",
                    "v1",
                    Some("b".repeat(64)),
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap();

        let error = state
            .append_final_transcript_revision(revision)
            .unwrap_err();
        assert!(error.contains("must be persisted from an ASR response"));
        assert!(state.list_timeline(Some(session.id)).unwrap().is_empty());
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn creates_lists_and_renames_anonymous_speaker_clusters() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();

        assert!(state
            .list_speaker_clusters(Some(session.id))
            .unwrap()
            .is_empty());
        let created = state.create_speaker_cluster(session.id).unwrap();
        assert!(created.updated_spans.is_empty());
        assert_eq!(created.clusters.len(), 1);
        let cluster = created.clusters[0].clone();
        assert_eq!(cluster.session_id, session.id);
        assert_eq!(cluster.label, "Speaker 1");
        assert!(!cluster.is_user_named);
        assert_eq!(cluster.label_revision, 1);
        assert_eq!(state.list_speaker_clusters(None).unwrap(), created.clusters);

        let renamed = state
            .rename_speaker_cluster(
                session.id,
                cluster.id.clone(),
                cluster.label_revision,
                "  会议主持人  ".to_owned(),
            )
            .unwrap();
        assert!(renamed.updated_spans.is_empty());
        assert_eq!(renamed.clusters.len(), 1);
        assert_eq!(renamed.clusters[0].label, "会议主持人");
        assert!(renamed.clusters[0].is_user_named);
        assert_eq!(renamed.clusters[0].label_revision, 2);

        let event_count = state.audit_trail.lock().unwrap().events().len();
        let label_revisions = state
            .audit_store
            .lock()
            .unwrap()
            .get_latest_speaker_cluster_label_revision(&cluster.id)
            .unwrap()
            .unwrap();
        assert_eq!(label_revisions.revision, 2);
        assert!(state
            .rename_speaker_cluster(session.id, cluster.id, 1, "过期名称".to_owned())
            .unwrap_err()
            .contains("changed"));
        assert_eq!(
            state.audit_trail.lock().unwrap().events().len(),
            event_count
        );
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn rejects_speaker_cluster_creation_for_an_unknown_session_without_audit_writes() {
        let state = AppState::in_memory();
        let unknown_session_id = Uuid::new_v4();
        let event_count = state.audit_trail.lock().unwrap().events().len();

        assert_eq!(
            state
                .create_speaker_cluster(unknown_session_id)
                .unwrap_err(),
            "speaker cluster session does not exist"
        );
        assert_eq!(
            state.audit_trail.lock().unwrap().events().len(),
            event_count
        );
        assert!(state
            .audit_store
            .lock()
            .unwrap()
            .list_speaker_clusters(unknown_session_id)
            .unwrap()
            .is_empty());
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn reassigns_one_current_final_span_without_mutating_its_evidence() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let cluster = state
            .create_speaker_cluster(session.id)
            .unwrap()
            .clusters
            .remove(0);
        let wall_clock_start = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(40);
        let model =
            TranscriptModelProvenance::new("fixture", "fixture-asr", "v1", Some("a".repeat(64)))
                .unwrap();
        let original = TranscriptRevision::original(
            session.id,
            TranscriptTiming::new(
                session.started_monotonic_ns + 10,
                session.started_monotonic_ns + 90,
                wall_clock_start,
                wall_clock_start + Duration::milliseconds(80),
            )
            .unwrap(),
            None,
            "不改变原文的说话人修正。",
            true,
            TranscriptSource::Synthetic,
            Some(model),
            Some(0.73),
        )
        .unwrap();
        state
            .append_final_transcript_revision(original.clone())
            .unwrap();
        let event_count_before = state.audit_trail.lock().unwrap().events().len();

        let result = state
            .reassign_transcript_speaker(
                session.id,
                original.logical_span_id,
                original.revision,
                Some(cluster.id.clone()),
            )
            .unwrap();
        assert_eq!(
            result.updated_spans,
            vec![SpeakerSpanRef {
                id: original.logical_span_id,
                revision: 2,
            }]
        );
        assert_eq!(
            result
                .clusters
                .iter()
                .find(|item| item.id == cluster.id)
                .unwrap()
                .span_count,
            1
        );

        let revisions = state
            .audit_store
            .lock()
            .unwrap()
            .list_transcript_revisions(session.id)
            .unwrap();
        assert_eq!(revisions.len(), 2);
        let reassigned = revisions.last().unwrap();
        assert_eq!(reassigned.logical_span_id, original.logical_span_id);
        assert_eq!(reassigned.parent_revision_id, Some(original.id));
        assert_eq!(reassigned.revision, 2);
        assert_eq!(reassigned.source, TranscriptSource::UserEdited);
        assert_eq!(reassigned.speaker_cluster_id, Some(cluster.id.clone()));
        assert_eq!(reassigned.text, original.text);
        assert_eq!(reassigned.timing(), original.timing());
        assert_eq!(reassigned.model, original.model);
        assert_eq!(reassigned.confidence, original.confidence);
        drop(revisions);

        let timeline = state.list_timeline(Some(session.id)).unwrap();
        assert_eq!(timeline.len(), 1);
        assert_eq!(timeline[0].id, original.logical_span_id);
        assert_eq!(timeline[0].revision, 2);
        assert_eq!(timeline[0].speaker_cluster_id, Some(cluster.id));
        assert_eq!(timeline[0].text, original.text);

        let trail = state.audit_trail.lock().unwrap();
        assert_eq!(trail.events().len(), event_count_before + 1);
        let event = trail.events().last().unwrap();
        assert_eq!(event.kind, AuditKind::TranscriptSpeakerReassigned);
        assert_eq!(event.run_id, Some(session.id));
        assert_eq!(event.causation_id, Some(original.id));
        assert!(event.wall_clock > original.wall_clock_end);
        drop(trail);
        assert!(state.audit_is_valid().unwrap());
    }

    #[test]
    fn rejects_stale_partial_unknown_cross_session_and_aliased_reassignment_targets() {
        let state = AppState::in_memory();
        let session = state.start_session().unwrap();
        let active_cluster = state
            .create_speaker_cluster(session.id)
            .unwrap()
            .clusters
            .remove(0);
        let aliased_cluster = state
            .create_speaker_cluster(session.id)
            .unwrap()
            .clusters
            .into_iter()
            .find(|cluster| cluster.id != active_cluster.id)
            .unwrap();
        let wall_clock_start = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(60);
        let original = TranscriptRevision::original(
            session.id,
            TranscriptTiming::new(
                session.started_monotonic_ns + 10,
                session.started_monotonic_ns + 20,
                wall_clock_start,
                wall_clock_start + Duration::nanoseconds(10),
            )
            .unwrap(),
            None,
            "只允许修正最终版本。",
            true,
            TranscriptSource::Synthetic,
            None,
            None,
        )
        .unwrap();
        state
            .append_final_transcript_revision(original.clone())
            .unwrap();
        let partial = TranscriptSpan::new(
            session.id,
            original.capture_start_ns,
            original.capture_end_ns,
            None,
            "临时片段。",
            false,
            1,
            TranscriptSource::Synthetic,
        )
        .unwrap();
        state.append_transcript(partial.clone()).unwrap();

        state.stop_session().unwrap();
        let foreign_session = state.start_session().unwrap();
        let foreign_cluster = state
            .create_speaker_cluster(foreign_session.id)
            .unwrap()
            .clusters
            .remove(0);
        append_test_speaker_alias(&state, session.id, &aliased_cluster.id, &active_cluster.id);
        let before = speaker_reassignment_snapshot(&state, session.id);

        assert!(state
            .reassign_transcript_speaker(
                session.id,
                original.logical_span_id,
                original.revision + 1,
                Some(active_cluster.id.clone()),
            )
            .is_err());
        assert_eq!(speaker_reassignment_snapshot(&state, session.id), before);

        assert!(state
            .reassign_transcript_speaker(
                session.id,
                partial.id,
                partial.revision,
                Some(active_cluster.id.clone()),
            )
            .is_err());
        assert_eq!(speaker_reassignment_snapshot(&state, session.id), before);

        assert!(state
            .reassign_transcript_speaker(
                session.id,
                Uuid::new_v4(),
                1,
                Some(active_cluster.id.clone()),
            )
            .is_err());
        assert_eq!(speaker_reassignment_snapshot(&state, session.id), before);

        assert!(state
            .reassign_transcript_speaker(
                session.id,
                original.logical_span_id,
                original.revision,
                Some(foreign_cluster.id),
            )
            .is_err());
        assert_eq!(speaker_reassignment_snapshot(&state, session.id), before);

        assert!(state
            .reassign_transcript_speaker(
                session.id,
                original.logical_span_id,
                original.revision,
                Some(aliased_cluster.id),
            )
            .is_err());
        assert_eq!(speaker_reassignment_snapshot(&state, session.id), before);
        assert!(state.audit_is_valid().unwrap());
    }

    fn append_test_speaker_alias(
        state: &AppState,
        session_id: Uuid,
        source_cluster_id: &str,
        target_cluster_id: &str,
    ) {
        let alias = SpeakerClusterAliasRevision::aliased_to(
            source_cluster_id.to_owned(),
            target_cluster_id.to_owned(),
        )
        .unwrap();
        let mut trail = state.audit_trail.lock().unwrap();
        let event = trail
            .next_event(
                Some(session_id),
                alias.parent_revision_id,
                AuditKind::SpeakerClusterAliasRevisionRecorded,
                state.monotonic_ns(),
                Utc::now(),
                &alias,
            )
            .unwrap();
        state
            .audit_store
            .lock()
            .unwrap()
            .append_speaker_cluster_alias_revision_with_audit(&event, &alias)
            .unwrap();
        assert!(trail.append_event(event));
    }

    fn speaker_reassignment_snapshot(
        state: &AppState,
        session_id: Uuid,
    ) -> (Vec<TranscriptSpan>, usize, usize) {
        let timeline = state.list_timeline(Some(session_id)).unwrap();
        let revision_count = state
            .audit_store
            .lock()
            .unwrap()
            .list_transcript_revisions(session_id)
            .unwrap()
            .len();
        let audit_event_count = state.audit_trail.lock().unwrap().events().len();
        (timeline, revision_count, audit_event_count)
    }

    #[test]
    fn rebuilds_the_final_timeline_projection_when_reopened() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-reopen-transcript-{}.sqlite3",
            Uuid::new_v4()
        ));
        let revision = {
            let state = AppState::open(&database).unwrap();
            let session = state.start_session().unwrap();
            let revision = TranscriptRevision::original(
                session.id,
                TranscriptTiming::new(
                    session.started_monotonic_ns,
                    session.started_monotonic_ns + 1_000_000_000,
                    session.started_at,
                    session.started_at + Duration::seconds(1),
                )
                .unwrap(),
                None,
                "重启后仍可检索的本地记录。",
                true,
                TranscriptSource::UserEdited,
                Some(
                    TranscriptModelProvenance::new(
                        "fixture",
                        "fixture-asr",
                        "v1",
                        Some("b".repeat(64)),
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap();
            state
                .append_final_transcript_revision(revision.clone())
                .unwrap();
            revision
        };

        let reopened = AppState::open(&database).unwrap();
        let timeline = reopened.list_timeline(Some(revision.session_id)).unwrap();
        assert_eq!(timeline.len(), 1);
        assert_eq!(timeline[0].id, revision.logical_span_id);
        assert_eq!(timeline[0].text, "重启后仍可检索的本地记录。");
        assert!(reopened.audit_is_valid().unwrap());
        drop(reopened);

        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute_batch("DROP TRIGGER transcript_revisions_are_immutable_update;")
            .unwrap();
        connection
            .execute(
                "UPDATE transcript_revisions SET text = ?1 WHERE id = ?2",
                rusqlite::params!["篡改后的记录。", revision.id.to_string()],
            )
            .unwrap();
        assert!(matches!(
            AppState::open(&database),
            Err(AuditStoreError::Integrity)
        ));
        drop(connection);

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn reopening_selects_the_latest_persisted_session_with_its_wall_clock_time() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-reopen-latest-session-{}.sqlite3",
            Uuid::new_v4()
        ));
        let first_wall_clock = DateTime::<Utc>::UNIX_EPOCH + Duration::hours(1);
        let second_wall_clock = DateTime::<Utc>::UNIX_EPOCH + Duration::hours(2);

        let (first_session_id, second_revision) = {
            let state = AppState::open(&database).unwrap();
            let first_session = state.start_session().unwrap();
            let first_revision = TranscriptRevision::original(
                first_session.id,
                TranscriptTiming::new(
                    first_session.started_monotonic_ns,
                    first_session.started_monotonic_ns + 1_000_000_000,
                    first_wall_clock,
                    first_wall_clock + Duration::seconds(1),
                )
                .unwrap(),
                None,
                "较早的本地记录。",
                true,
                TranscriptSource::UserEdited,
                Some(
                    TranscriptModelProvenance::new(
                        "fixture",
                        "fixture-asr",
                        "v1",
                        Some("c".repeat(64)),
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap();
            state
                .append_final_transcript_revision(first_revision)
                .unwrap();
            state.stop_session().unwrap();

            let second_session = state.start_session().unwrap();
            let second_revision = TranscriptRevision::original(
                second_session.id,
                TranscriptTiming::new(
                    second_session.started_monotonic_ns,
                    second_session.started_monotonic_ns + 1_000_000_000,
                    second_wall_clock,
                    second_wall_clock + Duration::seconds(1),
                )
                .unwrap(),
                None,
                "最近的本地记录。",
                true,
                TranscriptSource::UserEdited,
                Some(
                    TranscriptModelProvenance::new(
                        "fixture",
                        "fixture-asr",
                        "v1",
                        Some("d".repeat(64)),
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap();
            state
                .append_final_transcript_revision(second_revision.clone())
                .unwrap();
            (first_session.id, second_revision)
        };

        let reopened = AppState::open(&database).unwrap();
        let latest = reopened.list_timeline(None).unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0].session_id, second_revision.session_id);
        assert_eq!(latest[0].text, "最近的本地记录。");
        assert_eq!(latest[0].wall_clock_start, Some(second_wall_clock));
        assert_eq!(
            reopened
                .list_timeline(Some(first_session_id))
                .unwrap()
                .len(),
            1
        );
        assert!(reopened.audit_is_valid().unwrap());

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn imports_and_reopens_an_audited_local_model_without_egress() {
        let directory = std::env::temp_dir().join(format!(
            "word-covenant-state-model-import-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let database = directory.join("word-covenant.sqlite3");
        let source_path = directory.join("fixture-model.ggml");
        let bytes = b"local model fixture bytes";
        std::fs::write(&source_path, bytes).unwrap();
        let expected_sha256 = format!("{:x}", Sha256::digest(bytes));
        let request = ModelImportRequest {
            id: Uuid::new_v4(),
            source_path,
            model_kind: LocalModelKind::SpeechRecognition,
            version: "fixture-v1".to_owned(),
            input_format: WHISPER_CPP_GGML_INPUT_FORMAT.to_owned(),
            expected_sha256,
            license_acknowledgement: Some(
                LicenseAcknowledgement::new(
                    "word-covenant/fixture-model",
                    "test-license",
                    DateTime::<Utc>::UNIX_EPOCH,
                )
                .unwrap(),
            ),
        };

        let imported = {
            let state = AppState::open(&database).unwrap();
            let imported = state.import_local_model(request).unwrap();
            assert_eq!(state.list_local_models().unwrap(), vec![imported.clone()]);
            assert!(!imported.file_path.is_absolute());
            assert!(directory.join("models").join(&imported.file_path).exists());
            assert!(state.audit_is_valid().unwrap());
            imported
        };

        let reopened = AppState::open(&database).unwrap();
        assert_eq!(reopened.list_local_models().unwrap(), vec![imported]);
        assert!(reopened.audit_is_valid().unwrap());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn local_asr_profile_requires_an_explicit_compatible_model() {
        let directory = std::env::temp_dir().join(format!(
            "word-covenant-active-asr-profile-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let database = directory.join("word-covenant.sqlite3");
        let state = AppState::open(&database).unwrap();

        assert_eq!(state.active_local_asr_profile().unwrap(), None);
        assert!(state
            .select_active_local_asr_model(Uuid::new_v4())
            .unwrap_err()
            .contains("not registered"));

        let import = |kind: LocalModelKind, input_format: &str| {
            let id = Uuid::new_v4();
            let source_path = directory.join(format!("fixture-{id}.model"));
            let bytes = format!("local model fixture {id}").into_bytes();
            std::fs::write(&source_path, &bytes).unwrap();
            state
                .import_local_model(ModelImportRequest {
                    id,
                    source_path,
                    model_kind: kind,
                    version: "fixture-v1".to_owned(),
                    input_format: input_format.to_owned(),
                    expected_sha256: format!("{:x}", Sha256::digest(&bytes)),
                    license_acknowledgement: Some(
                        LicenseAcknowledgement::new(
                            "word-covenant/fixture-model",
                            "test-license",
                            DateTime::<Utc>::UNIX_EPOCH,
                        )
                        .unwrap(),
                    ),
                })
                .unwrap()
        };

        let vad_model = import(
            LocalModelKind::VoiceActivityDetection,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        assert!(state
            .select_active_local_asr_model(vad_model.id)
            .unwrap_err()
            .contains("not a speech-recognition model"));

        let incompatible_asr = import(LocalModelKind::SpeechRecognition, "gguf");
        assert!(state
            .select_active_local_asr_model(incompatible_asr.id)
            .unwrap_err()
            .contains("not compatible"));

        let selected_model = import(
            LocalModelKind::SpeechRecognition,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let profile = state
            .select_active_local_asr_model(selected_model.id)
            .unwrap();
        assert_eq!(profile.model_id, selected_model.id);
        assert_eq!(
            serde_json::to_value(&profile).unwrap(),
            serde_json::json!({ "modelId": selected_model.id })
        );

        // Test builds intentionally use the separate development mock path;
        // selecting a real ASR model remains forbidden while it is recording.
        let session = state.start_session().unwrap();
        assert!(state
            .select_active_local_asr_model(selected_model.id)
            .unwrap_err()
            .contains("while recording"));
        assert_eq!(state.stop_session().unwrap().unwrap().id, session.id);

        drop(state);
        let reopened = AppState::open(&database).unwrap();
        assert_eq!(reopened.active_local_asr_profile().unwrap(), None);
        assert!(reopened.audit_is_valid().unwrap());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn master_egress_gate_requires_an_enabled_session_and_matching_approval() {
        let state = AppState::in_memory();
        let categories = BTreeSet::from([DataCategory::Summary]);

        let startup_status = state.privacy_status().unwrap();
        assert!(startup_status.local_only);
        assert!(!startup_status.egress_enabled);
        assert_eq!(
            serde_json::to_value(startup_status).unwrap()["egressEnabled"],
            false
        );

        assert_eq!(
            state
                .evaluate_http_profile(
                    "crm-sync".to_owned(),
                    "https://api.example.com".to_owned(),
                    categories.clone(),
                )
                .unwrap(),
            PolicyDecision::Denied {
                reason: PolicyReason::EgressDisabled,
            }
        );

        let approval = state
            .create_egress_approval(
                "crm-sync".to_owned(),
                "https://api.example.com".to_owned(),
                categories.clone(),
                None,
            )
            .unwrap();
        let disabled_status_with_approval = state.privacy_status().unwrap();
        assert_eq!(disabled_status_with_approval.active_egress_approvals, 1);
        assert!(disabled_status_with_approval.local_only);
        assert!(!disabled_status_with_approval.egress_enabled);
        assert_eq!(
            state
                .evaluate_http_profile(
                    "crm-sync".to_owned(),
                    "https://api.example.com/path".to_owned(),
                    categories.clone(),
                )
                .unwrap(),
            PolicyDecision::Denied {
                reason: PolicyReason::EgressDisabled,
            }
        );

        let enabled_status = state.set_egress_enabled(true).unwrap();
        assert!(enabled_status.egress_enabled);
        assert!(!enabled_status.local_only);
        assert!(matches!(
            state
                .evaluate_http_profile(
                    "crm-sync".to_owned(),
                    "https://api.example.com/path".to_owned(),
                    categories.clone(),
                )
                .unwrap(),
            PolicyDecision::Allowed { .. }
        ));

        let disabled_status = state.set_egress_enabled(false).unwrap();
        assert!(!disabled_status.egress_enabled);
        assert!(disabled_status.local_only);
        assert_eq!(
            state
                .evaluate_http_profile(
                    "crm-sync".to_owned(),
                    "https://api.example.com/path".to_owned(),
                    categories,
                )
                .unwrap(),
            PolicyDecision::Denied {
                reason: PolicyReason::EgressDisabled,
            }
        );
        assert!(state.revoke_egress_approval(approval.id).unwrap());
        assert!(state.audit_is_valid().unwrap());
        assert!(state
            .audit_trail
            .lock()
            .unwrap()
            .events()
            .iter()
            .any(|event| matches!(event.kind, AuditKind::EgressSettingChanged)));
    }

    #[test]
    fn egress_switch_is_not_restored_when_app_state_reopens() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-egress-switch-{}.sqlite3",
            Uuid::new_v4()
        ));

        {
            let state = AppState::open(&database).unwrap();
            assert!(!state.privacy_status().unwrap().egress_enabled);
            assert!(state.set_egress_enabled(true).unwrap().egress_enabled);
            assert!(state.audit_is_valid().unwrap());
        }

        {
            let reopened = AppState::open(&database).unwrap();
            let status = reopened.privacy_status().unwrap();
            assert!(!status.egress_enabled);
            assert!(status.local_only);
            assert!(reopened.audit_is_valid().unwrap());
        }

        std::fs::remove_file(database).unwrap();
    }
}
