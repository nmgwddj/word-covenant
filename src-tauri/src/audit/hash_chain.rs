use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditKind {
    SessionStarted,
    SessionStopped,
    TranscriptRecorded,
    PlanProposed,
    EgressApprovalCreated,
    EgressApprovalRevoked,
    EgressSettingChanged,
    ActionDenied,
    ActionExecuted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditEvent {
    pub id: Uuid,
    pub run_id: Option<Uuid>,
    pub causation_id: Option<Uuid>,
    pub kind: AuditKind,
    pub monotonic_ns: u64,
    pub wall_clock: DateTime<Utc>,
    pub payload_hash: String,
    pub previous_hash: Option<String>,
    pub hash: String,
}

impl AuditEvent {
    pub fn new<T: Serialize>(
        run_id: Option<Uuid>,
        causation_id: Option<Uuid>,
        kind: AuditKind,
        monotonic_ns: u64,
        wall_clock: DateTime<Utc>,
        payload: &T,
        previous_hash: Option<String>,
    ) -> Result<Self, serde_json::Error> {
        let payload_hash = hash_bytes(&serde_json::to_vec(payload)?);
        let id = Uuid::new_v4();
        let hash = hash_event(
            id,
            run_id,
            causation_id,
            &kind,
            monotonic_ns,
            wall_clock,
            &payload_hash,
            previous_hash.as_deref(),
        );

        Ok(Self {
            id,
            run_id,
            causation_id,
            kind,
            monotonic_ns,
            wall_clock,
            payload_hash,
            previous_hash,
            hash,
        })
    }

    pub fn verifies(&self) -> bool {
        self.hash
            == hash_event(
                self.id,
                self.run_id,
                self.causation_id,
                &self.kind,
                self.monotonic_ns,
                self.wall_clock,
                &self.payload_hash,
                self.previous_hash.as_deref(),
            )
    }
}

#[derive(Clone, Debug, Default)]
pub struct AuditTrail {
    events: Vec<AuditEvent>,
}

impl AuditTrail {
    pub fn from_events(events: Vec<AuditEvent>) -> Self {
        Self { events }
    }

    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    pub fn next_event<T: Serialize>(
        &self,
        run_id: Option<Uuid>,
        causation_id: Option<Uuid>,
        kind: AuditKind,
        monotonic_ns: u64,
        wall_clock: DateTime<Utc>,
        payload: &T,
    ) -> Result<AuditEvent, serde_json::Error> {
        AuditEvent::new(
            run_id,
            causation_id,
            kind,
            monotonic_ns,
            wall_clock,
            payload,
            self.events.last().map(|event| event.hash.clone()),
        )
    }

    pub fn append_event(&mut self, event: AuditEvent) -> bool {
        let expected_previous = self.events.last().map(|previous| previous.hash.as_str());
        if event.previous_hash.as_deref() != expected_previous || !event.verifies() {
            return false;
        }
        self.events.push(event);
        true
    }

    pub fn append<T: Serialize>(
        &mut self,
        run_id: Option<Uuid>,
        causation_id: Option<Uuid>,
        kind: AuditKind,
        monotonic_ns: u64,
        wall_clock: DateTime<Utc>,
        payload: &T,
    ) -> Result<&AuditEvent, serde_json::Error> {
        let event = self.next_event(
            run_id,
            causation_id,
            kind,
            monotonic_ns,
            wall_clock,
            payload,
        )?;
        debug_assert!(self.append_event(event));
        Ok(self.events.last().expect("audit event was just appended"))
    }

    pub fn verify(&self) -> bool {
        self.events.iter().enumerate().all(|(index, event)| {
            let expected_previous = index
                .checked_sub(1)
                .and_then(|previous| self.events.get(previous))
                .map(|previous| previous.hash.as_str());

            event.previous_hash.as_deref() == expected_previous && event.verifies()
        })
    }
}

fn hash_event(
    id: Uuid,
    run_id: Option<Uuid>,
    causation_id: Option<Uuid>,
    kind: &AuditKind,
    monotonic_ns: u64,
    wall_clock: DateTime<Utc>,
    payload_hash: &str,
    previous_hash: Option<&str>,
) -> String {
    let material = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        id,
        run_id.map_or_else(String::new, |value| value.to_string()),
        causation_id.map_or_else(String::new, |value| value.to_string()),
        serde_json::to_string(kind).expect("audit kinds are serializable"),
        monotonic_ns,
        wall_clock.to_rfc3339(),
        payload_hash,
        previous_hash.unwrap_or_default()
    );
    hash_bytes(material.as_bytes())
}

fn hash_bytes(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_events_without_retaining_the_original_payload() {
        let mut trail = AuditTrail::default();
        trail
            .append(
                None,
                None,
                AuditKind::SessionStarted,
                10,
                Utc::now(),
                &serde_json::json!({ "rawText": "private spoken words" }),
            )
            .unwrap();
        trail
            .append(
                None,
                None,
                AuditKind::ActionDenied,
                20,
                Utc::now(),
                &serde_json::json!({ "reason": "denied_by_default" }),
            )
            .unwrap();

        assert!(trail.verify());
        assert_ne!(trail.events()[0].hash, trail.events()[1].hash);
        assert_eq!(
            trail.events()[1].previous_hash.as_deref(),
            Some(trail.events()[0].hash.as_str())
        );
        assert!(!serde_json::to_string(trail.events())
            .unwrap()
            .contains("private spoken words"));
    }

    #[test]
    fn detects_a_tampered_event() {
        let mut trail = AuditTrail::default();
        trail
            .append(
                None,
                None,
                AuditKind::SessionStarted,
                10,
                Utc::now(),
                &serde_json::json!({ "session": "one" }),
            )
            .unwrap();
        trail.events[0].payload_hash.push('0');

        assert!(!trail.verify());
    }
}
