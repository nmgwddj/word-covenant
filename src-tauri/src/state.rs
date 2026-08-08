#[cfg(all(target_os = "macos", not(test)))]
use crate::audio::CaptureStart;
#[cfg(target_os = "macos")]
use crate::audio::{CaptureGap, CaptureProjection, CaptureService};
#[cfg(any(test, debug_assertions))]
use crate::audio::{DevelopmentMockProgress, DevelopmentMockRunner};
use crate::audit::{AuditKind, AuditStore, AuditStoreError, AuditTrail};
#[cfg(target_os = "macos")]
use crate::domain::CaptureSegment;
use crate::domain::{
    CaptureSession, DataCategory, TranscriptModelProvenance, TranscriptRevision, TranscriptSource,
    TranscriptSpan, TranscriptTiming,
};
use crate::inference::model_registry::{ModelImportRequest, ModelRegistry, RegisteredModel};
use crate::inference::{
    AsrResponse, MappedTranscriptEmission, TranscriptEmission, TranscriptEmissionMapper,
};
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

pub struct AppState {
    started_at: Instant,
    sessions: Mutex<BTreeMap<Uuid, CaptureSession>>,
    timelines: Mutex<BTreeMap<Uuid, Vec<TranscriptSpan>>>,
    actions: Mutex<Vec<AgentAction>>,
    policy: Mutex<EgressPolicy>,
    audit_trail: Mutex<AuditTrail>,
    audit_store: Mutex<AuditStore>,
    model_registry: Mutex<ModelRegistry>,
    inference_mapper: Mutex<TranscriptEmissionMapper>,
    model_root: Option<PathBuf>,
    #[cfg(target_os = "macos")]
    capture_service: Mutex<CaptureService>,
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
            inference_mapper: Mutex::new(TranscriptEmissionMapper::default()),
            model_root,
            #[cfg(target_os = "macos")]
            capture_service: Mutex::new(CaptureService::new()),
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
        let (projection, gaps) = {
            let mut service = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?;
            let projection = service.projection();
            let gaps = service.take_pending_gaps();
            (projection, gaps)
        };
        if let Some(session) = self.active_recording_session()? {
            if let Err(error) = self.record_capture_gaps(session.id, gaps) {
                let _ = self.stop_session();
                return Err(error);
            }
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
        if let Some(active) = self.active_recording_session()? {
            return Ok(active);
        }

        let capture_start = self
            .capture_service
            .lock()
            .map_err(|_| "capture service lock poisoned".to_owned())?
            .start()?;
        let session = match self.start_session_at(capture_start.anchor.clone(), false) {
            Ok(session) => session,
            Err(error) => {
                let _ = self
                    .capture_service
                    .lock()
                    .map_err(|_| "capture service lock poisoned".to_owned())
                    .and_then(|mut service| service.stop().map(|_| ()));
                return Err(error);
            }
        };
        if let Err(error) = self.record_capture_started(&session, &capture_start) {
            let _ = self.stop_session();
            return Err(error);
        }
        Ok(session)
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

    #[cfg(all(target_os = "macos", not(test)))]
    fn record_capture_started(
        &self,
        session: &CaptureSession,
        capture_start: &CaptureStart,
    ) -> Result<(), String> {
        let segment = CaptureSegment::new(
            session.id,
            capture_start.device.uid(),
            capture_start.device.name(),
            capture_start.sample_rate,
            capture_start.channels,
            capture_start.anchor.monotonic_ns,
            capture_start.anchor.wall_clock,
        )?;
        self.record_capture_segment(&segment)?;
        self.record_audit(
            AuditKind::CaptureInputStarted,
            capture_start.anchor.monotonic_ns,
            &serde_json::json!({
                "sessionId": session.id,
                "deviceUid": capture_start.device.uid(),
                "deviceName": capture_start.device.name(),
                "sampleRate": capture_start.sample_rate,
                "channels": capture_start.channels,
                "anchor": capture_start.anchor,
            }),
        )
    }

    #[cfg(target_os = "macos")]
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
    fn record_capture_gaps(&self, session_id: Uuid, gaps: Vec<CaptureGap>) -> Result<(), String> {
        for gap in gaps {
            self.record_capture_gap(session_id, &gap)?;
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

    fn start_session_at(
        &self,
        point: crate::audio::CapturePoint,
        reject_active_session: bool,
    ) -> Result<CaptureSession, String> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| "session state lock poisoned".to_owned())?;

        if let Some(active) = sessions
            .values()
            .find(|session| matches!(session.state, crate::domain::SessionState::Recording))
        {
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
        let progress = {
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

        for span in progress.spans.iter().cloned() {
            self.append_transcript(span)?;
        }
        Ok(progress)
    }

    pub fn stop_session(&self) -> Result<Option<CaptureSession>, String> {
        #[cfg(all(target_os = "macos", not(test)))]
        let (stopped_native_input, pending_gaps) = {
            let mut service = self
                .capture_service
                .lock()
                .map_err(|_| "capture service lock poisoned".to_owned())?;
            let stopped_native_input = service.stop()?;
            let pending_gaps = service.take_pending_gaps();
            (stopped_native_input, pending_gaps)
        };
        #[cfg(any(not(target_os = "macos"), test))]
        let stopped_native_input = false;

        #[cfg(all(target_os = "macos", not(test)))]
        let active_session = self.active_recording_session()?;
        #[cfg(all(target_os = "macos", not(test)))]
        let capture_gap_result = active_session
            .as_ref()
            .map(|session| self.record_capture_gaps(session.id, pending_gaps))
            .transpose();

        let now = Utc::now();
        let monotonic_ns = self.monotonic_ns();
        let stopped = {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|_| "session state lock poisoned".to_owned())?;
            let Some(session) = sessions
                .values_mut()
                .find(|session| matches!(session.state, crate::domain::SessionState::Recording))
            else {
                return Ok(None);
            };

            session.stop(now);
            let stopped = session.clone();
            self.record_audit(AuditKind::SessionStopped, monotonic_ns, &stopped)?;
            stopped
        };

        #[cfg(any(test, debug_assertions))]
        self.stop_development_mock(stopped.id)?;

        #[cfg(target_os = "macos")]
        if stopped_native_input {
            self.record_audit(
                AuditKind::CaptureInputStopped,
                monotonic_ns,
                &serde_json::json!({ "sessionId": stopped.id }),
            )?;
        }

        #[cfg(all(target_os = "macos", not(test)))]
        capture_gap_result?;

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

    /// Persist a final inference/correction revision before exposing its compact
    /// timeline projection. Partial ASR output remains transient Rust/UI state
    /// and never enters Agent context through this path.
    pub fn append_final_transcript_revision(
        &self,
        revision: TranscriptRevision,
    ) -> Result<(), String> {
        if !revision.is_final {
            return Err(
                "only final transcript revisions may be persisted for Agent context".to_owned(),
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

        let mut mapper = self
            .inference_mapper
            .lock()
            .map_err(|_| "inference mapper lock poisoned".to_owned())?;
        let mut projections = Vec::new();
        for emission in response.emissions {
            let mapped = mapper
                .map(session_id, emission)
                .map_err(|error| format!("could not map local ASR emission: {error}"))?;
            let MappedTranscriptEmission::Final(final_emission) = mapped else {
                continue;
            };

            let persisted = (|| {
                let revision = transcript_revision_from_final_emission(
                    &session,
                    &final_emission.emission,
                    final_emission.logical_span_id,
                )?;
                let projection = transcript_revision_projection(&revision);
                self.append_final_transcript_revision(revision)?;
                Ok::<_, String>(projection)
            })();
            match persisted {
                Ok(projection) => {
                    // Keep the reservation pending until SQLite and its audit
                    // event have both committed, so concurrent/replayed finals
                    // cannot create a second durable revision.
                    mapper.commit_final(&final_emission);
                    projections.push(projection);
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

    pub fn list_local_models(&self) -> Result<Vec<RegisteredModel>, String> {
        let registry = self
            .model_registry
            .lock()
            .map_err(|_| "model registry lock poisoned".to_owned())?;
        Ok(registry.models().cloned().collect())
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
    fn stop_development_mock(&self, session_id: Uuid) -> Result<(), String> {
        let mut development_mock = self
            .development_mock
            .lock()
            .map_err(|_| "development mock state lock poisoned".to_owned())?;
        let Some(mut runner) = development_mock.take() else {
            return Ok(());
        };
        if runner.session_id() == session_id {
            runner.stop()
        } else {
            *development_mock = Some(runner);
            Ok(())
        }
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
        self.audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())
            .map(|trail| trail.verify())
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
        let mut trail = self
            .audit_trail
            .lock()
            .map_err(|_| "audit state lock poisoned".to_owned())?;
        let event = trail
            .next_event(None, None, kind, monotonic_ns, Utc::now(), payload)
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
) -> Result<TranscriptRevision, String> {
    let timing = TranscriptTiming::new(
        emission.capture_start_ns,
        emission.capture_end_ns,
        session_wall_clock_at(session, emission.capture_start_ns)?,
        session_wall_clock_at(session, emission.capture_end_ns)?,
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

fn session_wall_clock_at(
    session: &CaptureSession,
    capture_ns: u64,
) -> Result<DateTime<Utc>, String> {
    let offset_ns = capture_ns
        .checked_sub(session.started_monotonic_ns)
        .ok_or_else(|| "ASR emission begins before its capture session".to_owned())?;
    let offset_ns = i64::try_from(offset_ns)
        .map_err(|_| "ASR emission offset exceeds the supported wall-clock range".to_owned())?;
    Ok(session.started_at + chrono::Duration::nanoseconds(offset_ns))
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::audio::{CaptureGap, CaptureGapReason, CapturePoint};
    #[cfg(target_os = "macos")]
    use crate::domain::CaptureSegment;
    use crate::domain::{
        TranscriptModelProvenance, TranscriptRevision, TranscriptSource, TranscriptTiming,
    };
    use crate::inference::model_registry::{
        LicenseAcknowledgement, LocalModelKind, ModelImportRequest,
    };
    use crate::inference::{
        AsrEngine, AsrRequest, FixtureAsr, InferenceAudioWindow, INFERENCE_CHANNELS,
        INFERENCE_SAMPLE_RATE_HZ,
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
    fn development_mock_records_the_fixed_script_locally_and_stops_cleanly() {
        let state = AppState::in_memory();
        let before = state.privacy_status().unwrap();
        let session = state.start_development_mock_session().unwrap();
        let mut spans = Vec::new();

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
        assert!(spans.iter().all(
            |span| span.session_id == session.id && span.source == TranscriptSource::Synthetic
        ));
        assert_eq!(spans[0].capture_start_ns, session.started_monotonic_ns);
        assert_eq!(
            spans[0].capture_end_ns - spans[0].capture_start_ns,
            2_800_000_000
        );
        assert_eq!(spans[1].speaker_cluster_id.as_deref(), Some("speaker-2"));
        assert_eq!(spans[2].text, "先生成一份待确认的行动草案。");
        assert_eq!(state.list_timeline(Some(session.id)).unwrap().len(), 3);
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
                TranscriptSource::LocalInference,
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
                TranscriptSource::LocalInference,
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
        let source_path = directory.join("fixture-model.gguf");
        let bytes = b"local model fixture bytes";
        std::fs::write(&source_path, bytes).unwrap();
        let expected_sha256 = format!("{:x}", Sha256::digest(bytes));
        let request = ModelImportRequest {
            id: Uuid::new_v4(),
            source_path,
            model_kind: LocalModelKind::SpeechRecognition,
            version: "fixture-v1".to_owned(),
            input_format: "gguf".to_owned(),
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
