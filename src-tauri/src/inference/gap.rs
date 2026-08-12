use crate::audio::CapturePoint;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The native bridge location where a captured range reached a terminal
/// non-transcript outcome.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceGapStage {
    Dispatcher,
    JobQueue,
    Worker,
    ResultQueue,
    Shutdown,
}

/// A terminal cause distinct from a physical [`crate::audio::CaptureGap`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceGapReason {
    SegmenterFailed,
    JobQueueSaturated,
    ResultQueueSaturated,
    LocalEngineUnavailable,
    EngineFailed,
    StoppedBeforeInference,
}

/// A known capture range that reached local inference but cannot produce a
/// final transcript. The record contains no PCM samples or transcript text.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceGap {
    pub id: Uuid,
    pub session_id: Uuid,
    pub runtime_id: Uuid,
    pub capture_segment_id: Uuid,
    pub job_id: Option<Uuid>,
    pub started_at: CapturePoint,
    pub ended_at: CapturePoint,
    pub stage: InferenceGapStage,
    pub reason: InferenceGapReason,
}

impl InferenceGap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: Uuid,
        session_id: Uuid,
        runtime_id: Uuid,
        capture_segment_id: Uuid,
        job_id: Option<Uuid>,
        started_at: CapturePoint,
        ended_at: CapturePoint,
        stage: InferenceGapStage,
        reason: InferenceGapReason,
    ) -> Result<Self, String> {
        let gap = Self {
            id,
            session_id,
            runtime_id,
            capture_segment_id,
            job_id,
            started_at,
            ended_at,
            stage,
            reason,
        };
        gap.validate()?;
        Ok(gap)
    }

    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("gap ID", self.id),
            ("session ID", self.session_id),
            ("runtime ID", self.runtime_id),
            ("capture segment ID", self.capture_segment_id),
        ] {
            if value.is_nil() {
                return Err(format!("inference gap {field} must not be empty"));
            }
        }
        if self.job_id.is_some_and(|job_id| job_id.is_nil()) {
            return Err("inference gap job ID must not be empty".to_owned());
        }
        if self.ended_at.monotonic_ns < self.started_at.monotonic_ns {
            return Err("inference gap end must not precede its start".to_owned());
        }
        if self.ended_at.wall_clock < self.started_at.wall_clock {
            return Err("inference gap wall-clock end must not precede its start".to_owned());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CapturePoint;
    use chrono::{DateTime, Duration, Utc};
    use uuid::Uuid;

    fn point(monotonic_ns: u64, seconds: i64) -> CapturePoint {
        CapturePoint {
            monotonic_ns,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(seconds),
        }
    }

    fn gap() -> InferenceGap {
        InferenceGap::new(
            Uuid::from_u128(1),
            Uuid::from_u128(2),
            Uuid::from_u128(3),
            Uuid::from_u128(4),
            Some(Uuid::from_u128(5)),
            point(1_000, 1),
            point(2_000, 2),
            InferenceGapStage::JobQueue,
            InferenceGapReason::JobQueueSaturated,
        )
        .unwrap()
    }

    #[test]
    fn creates_a_stable_range_bearing_gap() {
        let gap = gap();

        assert_eq!(gap.id, Uuid::from_u128(1));
        assert_eq!(gap.session_id, Uuid::from_u128(2));
        assert_eq!(gap.runtime_id, Uuid::from_u128(3));
        assert_eq!(gap.capture_segment_id, Uuid::from_u128(4));
        assert_eq!(gap.job_id, Some(Uuid::from_u128(5)));
        assert_eq!(gap.stage, InferenceGapStage::JobQueue);
        assert_eq!(gap.reason, InferenceGapReason::JobQueueSaturated);
        assert!(gap.validate().is_ok());
    }

    #[test]
    fn rejects_inverted_capture_ranges_and_empty_identities() {
        let mut inverted = gap();
        inverted.ended_at = point(999, 2);
        assert!(inverted.validate().is_err());

        let mut inverted_wall_clock = gap();
        inverted_wall_clock.ended_at = point(2_000, 0);
        assert!(inverted_wall_clock.validate().is_err());

        let mut missing_identity = gap();
        missing_identity.runtime_id = Uuid::nil();
        assert!(missing_identity.validate().is_err());

        let mut missing_job_identity = gap();
        missing_job_identity.job_id = Some(Uuid::nil());
        assert!(missing_job_identity.validate().is_err());
    }

    #[test]
    fn rejects_unrecognised_serialized_stage_and_reason() {
        let mut serialized = serde_json::to_value(gap()).unwrap();
        serialized["stage"] = serde_json::Value::String("not_a_stage".to_owned());
        assert!(serde_json::from_value::<InferenceGap>(serialized).is_err());

        let mut serialized = serde_json::to_value(gap()).unwrap();
        serialized["reason"] = serde_json::Value::String("not_a_reason".to_owned());
        assert!(serde_json::from_value::<InferenceGap>(serialized).is_err());
    }
}
