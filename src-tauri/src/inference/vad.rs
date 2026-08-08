use super::{InferenceAudioWindow, InferenceEngine, InferenceError, ModelProvenance};
use serde::{Deserialize, Serialize};

pub const MAX_VAD_SEGMENTS_PER_REQUEST: usize = 128;

/// Native-only input for one bounded voice activity detection call.
///
/// This intentionally does not implement Serde because it owns PCM samples.
#[derive(Clone, Debug, PartialEq)]
pub struct VadRequest {
    pub audio: InferenceAudioWindow,
}

impl VadRequest {
    pub fn new(audio: InferenceAudioWindow) -> Result<Self, String> {
        let request = Self { audio };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.audio.validate()
    }
}

/// A detected speech range on the capture clock.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceActivitySegment {
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
    pub speech_probability: f32,
}

impl VoiceActivitySegment {
    pub fn validate(&self) -> Result<(), String> {
        if self.capture_end_ns <= self.capture_start_ns {
            return Err("voice activity segment end must follow its start".to_owned());
        }
        if !self.speech_probability.is_finite() || !(0.0..=1.0).contains(&self.speech_probability) {
            return Err("voice activity probability must be between zero and one".to_owned());
        }
        Ok(())
    }
}

/// Bounded local VAD output with the exact model provenance used.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VadResponse {
    pub segments: Vec<VoiceActivitySegment>,
    pub model_provenance: ModelProvenance,
}

impl VadResponse {
    pub fn new(
        request: &VadRequest,
        model_provenance: ModelProvenance,
        segments: Vec<VoiceActivitySegment>,
    ) -> Result<Self, String> {
        request.validate()?;
        model_provenance.validate()?;
        if segments.len() > MAX_VAD_SEGMENTS_PER_REQUEST {
            return Err(format!(
                "VAD response exceeds {MAX_VAD_SEGMENTS_PER_REQUEST} segments"
            ));
        }

        let mut previous_end = None;
        for segment in &segments {
            segment.validate()?;
            if segment.capture_start_ns < request.audio.capture_start_ns()
                || segment.capture_end_ns > request.audio.capture_end_ns()
            {
                return Err("VAD segment must remain inside its requested audio window".to_owned());
            }
            if previous_end.is_some_and(|end| segment.capture_start_ns < end) {
                return Err("VAD segments must be ordered and non-overlapping".to_owned());
            }
            previous_end = Some(segment.capture_end_ns);
        }

        Ok(Self {
            segments,
            model_provenance,
        })
    }
}

/// A local VAD adapter. No VAD implementation receives network authority from
/// this contract; it only receives native-memory audio and returns metadata.
pub trait VadEngine: InferenceEngine {
    fn detect_voice_activity(
        &mut self,
        request: &VadRequest,
    ) -> Result<VadResponse, InferenceError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceAudioWindow, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ};
    use uuid::Uuid;

    fn model() -> ModelProvenance {
        ModelProvenance::new("fixture", "fixture-vad", "v1", "b".repeat(64)).unwrap()
    }

    fn request() -> VadRequest {
        VadRequest::new(
            InferenceAudioWindow::new(
                Uuid::nil(),
                0,
                1_000_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.0; 16_000],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rejects_overlapping_vad_segments() {
        let error = VadResponse::new(
            &request(),
            model(),
            vec![
                VoiceActivitySegment {
                    capture_start_ns: 0,
                    capture_end_ns: 500_000_000,
                    speech_probability: 0.9,
                },
                VoiceActivitySegment {
                    capture_start_ns: 499_000_000,
                    capture_end_ns: 800_000_000,
                    speech_probability: 0.9,
                },
            ],
        )
        .unwrap_err();

        assert!(error.contains("non-overlapping"));
    }

    #[test]
    fn serializes_vad_metadata_without_pcm() {
        let response = VadResponse::new(
            &request(),
            model(),
            vec![VoiceActivitySegment {
                capture_start_ns: 0,
                capture_end_ns: 500_000_000,
                speech_probability: 0.9,
            }],
        )
        .unwrap();

        let serialized = serde_json::to_string(&response).unwrap();

        assert!(serialized.contains("speechProbability"));
        assert!(!serialized.contains("samples"));
    }
}
