use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Recording,
    Stopped,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSession {
    pub id: Uuid,
    pub started_at: DateTime<Utc>,
    pub started_monotonic_ns: u64,
    pub stopped_at: Option<DateTime<Utc>>,
    pub state: SessionState,
}

impl CaptureSession {
    pub fn begin(started_monotonic_ns: u64, started_at: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4(),
            started_at,
            started_monotonic_ns,
            stopped_at: None,
            state: SessionState::Recording,
        }
    }

    pub fn stop(&mut self, stopped_at: DateTime<Utc>) {
        self.stopped_at = Some(stopped_at);
        self.state = SessionState::Stopped;
    }
}
