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

/// Immutable metadata for one contiguous capture clock segment.
///
/// A session can contain several segments when the microphone stream is
/// restarted. This deliberately contains no PCM or transcript content.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSegment {
    pub id: Uuid,
    pub session_id: Uuid,
    pub device_uid: String,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub anchor_monotonic_ns: u64,
    pub anchor_wall_clock: DateTime<Utc>,
}

impl CaptureSegment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: Uuid,
        device_uid: impl Into<String>,
        device_name: impl Into<String>,
        sample_rate: u32,
        channels: u16,
        anchor_monotonic_ns: u64,
        anchor_wall_clock: DateTime<Utc>,
    ) -> Result<Self, String> {
        let device_uid = device_uid.into();
        let device_name = device_name.into();
        if device_uid.trim().is_empty() {
            return Err("capture segment device uid must not be empty".to_owned());
        }
        if device_name.trim().is_empty() {
            return Err("capture segment device name must not be empty".to_owned());
        }
        if sample_rate == 0 {
            return Err("capture segment sample rate must be greater than zero".to_owned());
        }
        if channels == 0 {
            return Err("capture segment channel count must be greater than zero".to_owned());
        }

        Ok(Self {
            id: Uuid::new_v4(),
            session_id,
            device_uid,
            device_name,
            sample_rate,
            channels,
            anchor_monotonic_ns,
            anchor_wall_clock,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_capture_segment_metadata() {
        let now = Utc::now();
        let session_id = Uuid::new_v4();

        assert!(
            CaptureSegment::new(session_id, "", "Built-in Microphone", 48_000, 1, 0, now).is_err()
        );
        assert!(CaptureSegment::new(session_id, "built-in", "", 48_000, 1, 0, now).is_err());
        assert!(
            CaptureSegment::new(session_id, "built-in", "Built-in Microphone", 0, 1, 0, now,)
                .is_err()
        );
        assert!(CaptureSegment::new(
            session_id,
            "built-in",
            "Built-in Microphone",
            48_000,
            0,
            0,
            now,
        )
        .is_err());
    }
}
