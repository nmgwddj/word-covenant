use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Starting,
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
        Self::begin_with_id(Uuid::new_v4(), started_monotonic_ns, started_at)
            .expect("newly generated capture session ID must not be nil")
    }

    pub fn begin_with_id(
        id: Uuid,
        started_monotonic_ns: u64,
        started_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        Self::begin_with_state(
            id,
            started_monotonic_ns,
            started_at,
            SessionState::Recording,
        )
    }

    /// Create a native capture session whose durable start bundle exists but
    /// whose dispatcher has not yet been armed. It must be published through
    /// [`Self::publish_recording`] before external callers can treat it as a
    /// recording session.
    pub fn begin_starting_with_id(
        id: Uuid,
        started_monotonic_ns: u64,
        started_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        Self::begin_with_state(id, started_monotonic_ns, started_at, SessionState::Starting)
    }

    fn begin_with_state(
        id: Uuid,
        started_monotonic_ns: u64,
        started_at: DateTime<Utc>,
        state: SessionState,
    ) -> Result<Self, String> {
        if id.is_nil() {
            return Err("capture session id must not be nil".to_owned());
        }

        Ok(Self {
            id,
            started_at,
            started_monotonic_ns,
            stopped_at: None,
            state,
        })
    }

    pub fn publish_recording(&mut self) -> Result<(), String> {
        if !matches!(self.state, SessionState::Starting) {
            return Err("only a starting capture session can become recording".to_owned());
        }
        self.state = SessionState::Recording;
        Ok(())
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
        Self::new_with_id(
            Uuid::new_v4(),
            session_id,
            device_uid,
            device_name,
            sample_rate,
            channels,
            anchor_monotonic_ns,
            anchor_wall_clock,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_id(
        id: Uuid,
        session_id: Uuid,
        device_uid: impl Into<String>,
        device_name: impl Into<String>,
        sample_rate: u32,
        channels: u16,
        anchor_monotonic_ns: u64,
        anchor_wall_clock: DateTime<Utc>,
    ) -> Result<Self, String> {
        if id.is_nil() {
            return Err("capture segment id must not be nil".to_owned());
        }

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
            id,
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

    #[test]
    fn begins_capture_session_with_a_supplied_stable_id() {
        let id = Uuid::new_v4();
        let started_at = DateTime::<Utc>::UNIX_EPOCH;

        let session = CaptureSession::begin_with_id(id, 123, started_at).unwrap();

        assert_eq!(session.id, id);
        assert_eq!(session.started_monotonic_ns, 123);
        assert_eq!(session.started_at, started_at);
        assert_eq!(session.stopped_at, None);
        assert_eq!(session.state, SessionState::Recording);
        assert!(CaptureSession::begin_with_id(Uuid::nil(), 123, started_at).is_err());
    }

    #[test]
    fn publishes_a_starting_native_session_only_once() {
        let id = Uuid::new_v4();
        let started_at = DateTime::<Utc>::UNIX_EPOCH;
        let mut session = CaptureSession::begin_starting_with_id(id, 123, started_at).unwrap();

        assert_eq!(session.state, SessionState::Starting);
        session.publish_recording().unwrap();
        assert_eq!(session.state, SessionState::Recording);
        assert!(session.publish_recording().is_err());
    }

    #[test]
    fn creates_capture_segment_with_a_supplied_stable_id() {
        let id = Uuid::new_v4();
        let session_id = Uuid::new_v4();
        let anchor_wall_clock = DateTime::<Utc>::UNIX_EPOCH;

        let segment = CaptureSegment::new_with_id(
            id,
            session_id,
            "built-in",
            "Built-in Microphone",
            48_000,
            1,
            456,
            anchor_wall_clock,
        )
        .unwrap();

        assert_eq!(segment.id, id);
        assert_eq!(segment.session_id, session_id);
        assert_eq!(segment.device_uid, "built-in");
        assert_eq!(segment.device_name, "Built-in Microphone");
        assert_eq!(segment.sample_rate, 48_000);
        assert_eq!(segment.channels, 1);
        assert_eq!(segment.anchor_monotonic_ns, 456);
        assert_eq!(segment.anchor_wall_clock, anchor_wall_clock);
        assert!(CaptureSegment::new_with_id(
            Uuid::nil(),
            session_id,
            "built-in",
            "Built-in Microphone",
            48_000,
            1,
            456,
            anchor_wall_clock,
        )
        .is_err());
    }
}
