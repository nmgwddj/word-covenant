#[cfg(any(test, debug_assertions))]
use crate::audio::{DevelopmentMockProgress, DevelopmentMockRunner};
use crate::audit::{AuditKind, AuditStore, AuditStoreError, AuditTrail};
use crate::domain::{CaptureSession, DataCategory, TranscriptSpan};
use crate::policy::{EgressApproval, EgressPolicy, EgressRequest, PolicyDecision, PolicyReason};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
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
    #[cfg(any(test, debug_assertions))]
    development_mock: Mutex<Option<DevelopmentMockRunner>>,
}

impl AppState {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AuditStoreError> {
        let audit_store = AuditStore::open_path(path)?;
        let audit_trail = AuditTrail::from_events(audit_store.list()?);
        if !audit_trail.verify() {
            return Err(AuditStoreError::Integrity);
        }

        Ok(Self::from_audit_store(audit_store, audit_trail))
    }

    #[cfg(test)]
    pub fn in_memory() -> Self {
        let audit_store = AuditStore::open_in_memory().expect("in-memory audit store opens");
        Self::from_audit_store(audit_store, AuditTrail::default())
    }

    fn from_audit_store(audit_store: AuditStore, audit_trail: AuditTrail) -> Self {
        Self {
            started_at: Instant::now(),
            sessions: Mutex::new(BTreeMap::new()),
            timelines: Mutex::new(BTreeMap::new()),
            actions: Mutex::new(Vec::new()),
            policy: Mutex::new(EgressPolicy::default()),
            audit_trail: Mutex::new(audit_trail),
            audit_store: Mutex::new(audit_store),
            #[cfg(any(test, debug_assertions))]
            development_mock: Mutex::new(None),
        }
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
        self.start_session_with_active(false)
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

    fn start_session_with_active(
        &self,
        reject_active_session: bool,
    ) -> Result<CaptureSession, String> {
        let now = Utc::now();
        let monotonic_ns = self.monotonic_ns();
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

        let session = CaptureSession::begin(monotonic_ns, now);
        self.record_audit(AuditKind::SessionStarted, monotonic_ns, &session)?;
        self.timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?
            .entry(session.id)
            .or_default();
        sessions.insert(session.id, session.clone());
        Ok(session)
    }

    pub fn stop_session(&self) -> Result<Option<CaptureSession>, String> {
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

        Ok(Some(stopped))
    }

    pub fn list_timeline(&self, session_id: Option<Uuid>) -> Result<Vec<TranscriptSpan>, String> {
        let timelines = self
            .timelines
            .lock()
            .map_err(|_| "timeline state lock poisoned".to_owned())?;
        let mut spans = match session_id {
            Some(session_id) => timelines.get(&session_id).cloned().unwrap_or_default(),
            None => timelines
                .values()
                .flat_map(|timeline| timeline.iter().cloned())
                .collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::TranscriptSource;

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
