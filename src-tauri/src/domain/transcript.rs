use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptSource {
    Synthetic,
    LocalInference,
    UserEdited,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerCluster {
    pub id: String,
    pub label: String,
    pub is_user_named: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSpan {
    pub id: Uuid,
    pub session_id: Uuid,
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
    /// Retained for projections rebuilt from durable revisions. Synthetic and
    /// live-only spans continue to use the active session's monotonic clock.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_clock_start: Option<DateTime<Utc>>,
    pub speaker_cluster_id: Option<String>,
    pub text: String,
    pub is_final: bool,
    pub revision: u32,
    pub source: TranscriptSource,
}

impl TranscriptSpan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: Uuid,
        capture_start_ns: u64,
        capture_end_ns: u64,
        speaker_cluster_id: Option<String>,
        text: impl Into<String>,
        is_final: bool,
        revision: u32,
        source: TranscriptSource,
    ) -> Result<Self, String> {
        let text = text.into();

        if capture_end_ns < capture_start_ns {
            return Err("capture end must not precede capture start".to_owned());
        }
        if text.trim().is_empty() {
            return Err("transcript text must not be empty".to_owned());
        }

        Ok(Self {
            id: Uuid::new_v4(),
            session_id,
            capture_start_ns,
            capture_end_ns,
            wall_clock_start: None,
            speaker_cluster_id,
            text,
            is_final,
            revision,
            source,
        })
    }

    /// Adds the capture wall-clock information required for durable storage
    /// without changing the legacy timeline projection used by synthetic input.
    pub fn into_revision(
        self,
        timing: TranscriptTiming,
        model: Option<TranscriptModelProvenance>,
        confidence: Option<f64>,
    ) -> Result<TranscriptRevision, String> {
        TranscriptRevision::from_span(self, timing, model, confidence)
    }
}

/// The model that produced (or last materially changed) a transcript revision.
///
/// The model registry will own file paths and license acknowledgements. A
/// transcript only stores stable local provenance that remains meaningful when
/// a registry entry is later removed or renamed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptModelProvenance {
    pub provider: String,
    pub model_id: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl TranscriptModelProvenance {
    pub fn new(
        provider: impl Into<String>,
        model_id: impl Into<String>,
        version: impl Into<String>,
        sha256: Option<String>,
    ) -> Result<Self, String> {
        let provenance = Self {
            provider: provider.into(),
            model_id: model_id.into(),
            version: version.into(),
            sha256,
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_non_empty("model provider", &self.provider)?;
        validate_non_empty("model id", &self.model_id)?;
        validate_non_empty("model version", &self.version)?;
        if let Some(sha256) = &self.sha256 {
            if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("model SHA-256 must be a 64-character hexadecimal digest".to_owned());
            }
        }
        Ok(())
    }
}

/// Capture timing resolved by the native audio clock, not browser presentation
/// time. Both wall-clock endpoints are stored so a revision remains useful if
/// the session's capture anchor becomes unavailable.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptTiming {
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
    pub wall_clock_start: DateTime<Utc>,
    pub wall_clock_end: DateTime<Utc>,
}

impl TranscriptTiming {
    pub fn new(
        capture_start_ns: u64,
        capture_end_ns: u64,
        wall_clock_start: DateTime<Utc>,
        wall_clock_end: DateTime<Utc>,
    ) -> Result<Self, String> {
        let timing = Self {
            capture_start_ns,
            capture_end_ns,
            wall_clock_start,
            wall_clock_end,
        };
        timing.validate()?;
        Ok(timing)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.capture_end_ns < self.capture_start_ns {
            return Err("capture end must not precede capture start".to_owned());
        }
        if self.wall_clock_end < self.wall_clock_start {
            return Err("wall-clock end must not precede wall-clock start".to_owned());
        }
        Ok(())
    }
}

/// One immutable version of a logical transcript span.
///
/// `id` identifies this physical revision. `logical_span_id` is stable for the
/// whole history, and `parent_revision_id` forms an append-only chain back to
/// its original row. Corrections and ASR refinements always create a new value.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptRevision {
    pub id: Uuid,
    pub logical_span_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_revision_id: Option<Uuid>,
    pub session_id: Uuid,
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
    pub wall_clock_start: DateTime<Utc>,
    pub wall_clock_end: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_cluster_id: Option<String>,
    pub text: String,
    pub is_final: bool,
    pub revision: u32,
    pub source: TranscriptSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<TranscriptModelProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

impl TranscriptRevision {
    #[allow(clippy::too_many_arguments)]
    pub fn original(
        session_id: Uuid,
        timing: TranscriptTiming,
        speaker_cluster_id: Option<String>,
        text: impl Into<String>,
        is_final: bool,
        source: TranscriptSource,
        model: Option<TranscriptModelProvenance>,
        confidence: Option<f64>,
    ) -> Result<Self, String> {
        Self::original_with_id(
            Uuid::new_v4(),
            session_id,
            timing,
            speaker_cluster_id,
            text,
            is_final,
            source,
            model,
            confidence,
        )
    }

    /// Creates the durable first revision using a caller-provided logical
    /// span ID. Native inference uses this to preserve one identity across
    /// transient partial emissions and the first persisted final result.
    #[allow(clippy::too_many_arguments)]
    pub fn original_with_id(
        id: Uuid,
        session_id: Uuid,
        timing: TranscriptTiming,
        speaker_cluster_id: Option<String>,
        text: impl Into<String>,
        is_final: bool,
        source: TranscriptSource,
        model: Option<TranscriptModelProvenance>,
        confidence: Option<f64>,
    ) -> Result<Self, String> {
        let revision = Self {
            id,
            logical_span_id: id,
            parent_revision_id: None,
            session_id,
            capture_start_ns: timing.capture_start_ns,
            capture_end_ns: timing.capture_end_ns,
            wall_clock_start: timing.wall_clock_start,
            wall_clock_end: timing.wall_clock_end,
            speaker_cluster_id,
            text: text.into(),
            is_final,
            revision: 1,
            source,
            model,
            confidence,
        };
        revision.validate()?;
        Ok(revision)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn revision_of(
        previous: &Self,
        timing: TranscriptTiming,
        speaker_cluster_id: Option<String>,
        text: impl Into<String>,
        is_final: bool,
        source: TranscriptSource,
        model: Option<TranscriptModelProvenance>,
        confidence: Option<f64>,
    ) -> Result<Self, String> {
        let revision = previous
            .revision
            .checked_add(1)
            .ok_or_else(|| "transcript revision number overflowed".to_owned())?;
        let next = Self {
            id: Uuid::new_v4(),
            logical_span_id: previous.logical_span_id,
            parent_revision_id: Some(previous.id),
            session_id: previous.session_id,
            capture_start_ns: timing.capture_start_ns,
            capture_end_ns: timing.capture_end_ns,
            wall_clock_start: timing.wall_clock_start,
            wall_clock_end: timing.wall_clock_end,
            speaker_cluster_id,
            text: text.into(),
            is_final,
            revision,
            source,
            model,
            confidence,
        };
        next.validate()?;
        Ok(next)
    }

    pub fn from_span(
        span: TranscriptSpan,
        timing: TranscriptTiming,
        model: Option<TranscriptModelProvenance>,
        confidence: Option<f64>,
    ) -> Result<Self, String> {
        let revision = Self {
            id: span.id,
            logical_span_id: span.id,
            parent_revision_id: None,
            session_id: span.session_id,
            capture_start_ns: timing.capture_start_ns,
            capture_end_ns: timing.capture_end_ns,
            wall_clock_start: timing.wall_clock_start,
            wall_clock_end: timing.wall_clock_end,
            speaker_cluster_id: span.speaker_cluster_id,
            text: span.text,
            is_final: span.is_final,
            revision: span.revision,
            source: span.source,
            model,
            confidence,
        };
        revision.validate()?;
        Ok(revision)
    }

    pub fn timing(&self) -> TranscriptTiming {
        TranscriptTiming {
            capture_start_ns: self.capture_start_ns,
            capture_end_ns: self.capture_end_ns,
            wall_clock_start: self.wall_clock_start,
            wall_clock_end: self.wall_clock_end,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.timing().validate()?;
        validate_non_empty("transcript text", &self.text)?;

        match self.parent_revision_id {
            None if self.logical_span_id != self.id => {
                return Err(
                    "an original transcript revision must use its own ID as logical span ID"
                        .to_owned(),
                );
            }
            Some(parent_revision_id) if parent_revision_id == self.id => {
                return Err("a transcript revision cannot parent itself".to_owned());
            }
            Some(_) if self.logical_span_id == self.id => {
                return Err(
                    "a transcript revision must retain the original logical span ID".to_owned(),
                );
            }
            _ => {}
        }

        if let Some(confidence) = self.confidence {
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(
                    "transcript confidence must be a finite value between 0 and 1".to_owned(),
                );
            }
        }

        if let Some(model) = &self.model {
            model.validate()?;
        }
        if matches!(self.source, TranscriptSource::LocalInference) && self.model.is_none() {
            return Err("local inference transcripts require model provenance".to_owned());
        }
        Ok(())
    }
}

fn validate_non_empty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn rejects_an_inverted_capture_range() {
        let result = TranscriptSpan::new(
            Uuid::new_v4(),
            20,
            10,
            None,
            "hello",
            true,
            0,
            TranscriptSource::Synthetic,
        );

        assert!(result.is_err());
    }

    #[test]
    fn chains_immutable_revisions_to_the_original_span() {
        let start = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(10);
        let original = TranscriptRevision::original(
            Uuid::new_v4(),
            TranscriptTiming::new(1_000, 2_000, start, start + Duration::seconds(1)).unwrap(),
            None,
            "local fixture",
            false,
            TranscriptSource::LocalInference,
            Some(
                TranscriptModelProvenance::new("whisper.cpp", "ggml-small", "1.7.4", None).unwrap(),
            ),
            Some(0.7),
        )
        .unwrap();
        let next = TranscriptRevision::revision_of(
            &original,
            TranscriptTiming::new(1_000, 2_000, start, start + Duration::seconds(1)).unwrap(),
            None,
            "local fixture finalized",
            true,
            TranscriptSource::LocalInference,
            original.model.clone(),
            Some(0.9),
        )
        .unwrap();

        assert_eq!(original.logical_span_id, original.id);
        assert_eq!(original.revision, 1);
        assert_eq!(next.logical_span_id, original.id);
        assert_eq!(next.parent_revision_id, Some(original.id));
        assert_eq!(next.revision, original.revision + 1);
        assert_ne!(next.id, original.id);
    }

    #[test]
    fn keeps_legacy_synthetic_span_construction_compatible() {
        let session_id = Uuid::new_v4();
        let legacy = TranscriptSpan::new(
            session_id,
            1_000,
            2_000,
            Some("speaker-1".to_owned()),
            "fixture transcript",
            true,
            1,
            TranscriptSource::Synthetic,
        )
        .unwrap();
        let revision = legacy
            .into_revision(
                TranscriptTiming::new(
                    1_000,
                    2_000,
                    DateTime::<Utc>::UNIX_EPOCH,
                    DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(1),
                )
                .unwrap(),
                None,
                None,
            )
            .unwrap();

        assert_eq!(revision.session_id, session_id);
        assert_eq!(revision.revision, 1);
        assert_eq!(revision.source, TranscriptSource::Synthetic);
    }

    #[test]
    fn rejects_invalid_confidence_and_missing_local_model_provenance() {
        let timing = TranscriptTiming::new(
            1,
            2,
            DateTime::<Utc>::UNIX_EPOCH,
            DateTime::<Utc>::UNIX_EPOCH,
        )
        .unwrap();
        assert!(TranscriptRevision::original(
            Uuid::new_v4(),
            timing.clone(),
            None,
            "fixture",
            true,
            TranscriptSource::LocalInference,
            None,
            Some(0.8),
        )
        .is_err());
        assert!(TranscriptRevision::original(
            Uuid::new_v4(),
            timing,
            None,
            "fixture",
            true,
            TranscriptSource::Synthetic,
            None,
            Some(1.1),
        )
        .is_err());
    }
}
