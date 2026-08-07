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
            speaker_cluster_id,
            text,
            is_final,
            revision,
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
