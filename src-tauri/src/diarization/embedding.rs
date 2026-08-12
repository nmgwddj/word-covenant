use crate::inference::{InferenceAudioWindow, InferenceEngine, InferenceError, ModelProvenance};

pub const MAX_SPEAKER_EMBEDDING_DIMENSIONS: usize = 2_048;
const MIN_SPEAKER_EMBEDDING_DIMENSIONS: usize = 2;
const MIN_VECTOR_NORM: f64 = 1e-12;

/// A bounded, L2-normalized vector in one explicit local model space.
///
/// This type intentionally has no Serde implementation. Persistence is owned
/// by the audited voice-profile store; vectors must not accidentally cross
/// Tauri IPC or enter generic logs and audit payloads.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerEmbedding {
    model: ModelProvenance,
    values: Box<[f32]>,
}

impl SpeakerEmbedding {
    pub fn new(model: ModelProvenance, values: Vec<f32>) -> Result<Self, String> {
        model.validate()?;
        if !(MIN_SPEAKER_EMBEDDING_DIMENSIONS..=MAX_SPEAKER_EMBEDDING_DIMENSIONS)
            .contains(&values.len())
        {
            return Err(format!(
                "speaker embedding dimensions must be between {MIN_SPEAKER_EMBEDDING_DIMENSIONS} and {MAX_SPEAKER_EMBEDDING_DIMENSIONS}"
            ));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err("speaker embedding values must be finite".to_owned());
        }

        let norm_squared = values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if norm_squared <= MIN_VECTOR_NORM {
            return Err("speaker embedding vector must have a non-zero norm".to_owned());
        }
        let norm = norm_squared.sqrt();
        let values = values
            .into_iter()
            .map(|value| (f64::from(value) / norm) as f32)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self { model, values })
    }

    pub(crate) fn from_normalized(
        model: ModelProvenance,
        values: Vec<f32>,
    ) -> Result<Self, String> {
        model.validate()?;
        if !(MIN_SPEAKER_EMBEDDING_DIMENSIONS..=MAX_SPEAKER_EMBEDDING_DIMENSIONS)
            .contains(&values.len())
        {
            return Err("persisted speaker embedding has invalid dimensions".to_owned());
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err("persisted speaker embedding values must be finite".to_owned());
        }
        let norm_squared = values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if (norm_squared - 1.0).abs() > 1e-4 {
            return Err("persisted speaker embedding must be L2 normalized".to_owned());
        }
        Ok(Self {
            model,
            values: values.into_boxed_slice(),
        })
    }

    pub fn model(&self) -> &ModelProvenance {
        &self.model
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.model == other.model && self.dimensions() == other.dimensions()
    }
}

/// Compact quality evidence for one speaker sample. Values are derived inside
/// the native process and carry no PCM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerSampleQuality {
    voiced_duration_ns: u64,
    voiced_ratio: f32,
    signal_quality: f32,
    overlap_probability: f32,
}

impl SpeakerSampleQuality {
    pub fn new(
        voiced_duration_ns: u64,
        voiced_ratio: f32,
        signal_quality: f32,
        overlap_probability: f32,
    ) -> Result<Self, String> {
        for (label, value) in [
            ("voiced ratio", voiced_ratio),
            ("signal quality", signal_quality),
            ("overlap probability", overlap_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "speaker sample {label} must be between zero and one"
                ));
            }
        }
        Ok(Self {
            voiced_duration_ns,
            voiced_ratio,
            signal_quality,
            overlap_probability,
        })
    }

    pub fn voiced_duration_ns(self) -> u64 {
        self.voiced_duration_ns
    }

    pub fn voiced_ratio(self) -> f32 {
        self.voiced_ratio
    }

    pub fn signal_quality(self) -> f32 {
        self.signal_quality
    }

    pub fn overlap_probability(self) -> f32 {
        self.overlap_probability
    }
}

/// A local-only adapter. Implementations receive bounded 16 kHz mono windows
/// and must never fetch a model or transmit audio.
pub trait SpeakerEmbeddingEngine: InferenceEngine {
    fn embed(
        &mut self,
        audio: &InferenceAudioWindow,
    ) -> Result<(SpeakerEmbedding, SpeakerSampleQuality), InferenceError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(version: &str) -> ModelProvenance {
        ModelProvenance::new("fixture", "speaker-embedding", version, "a".repeat(64)).unwrap()
    }

    #[test]
    fn normalizes_a_finite_bounded_embedding_and_tracks_model_space() {
        let embedding = SpeakerEmbedding::new(model("v1"), vec![3.0, 4.0]).unwrap();

        assert_eq!(embedding.dimensions(), 2);
        assert!((embedding.values()[0] - 0.6).abs() < 1e-6);
        assert!((embedding.values()[1] - 0.8).abs() < 1e-6);
        assert!(embedding.is_compatible_with(&embedding));
        assert!(!embedding
            .is_compatible_with(&SpeakerEmbedding::new(model("v2"), vec![3.0, 4.0]).unwrap()));
    }

    #[test]
    fn rejects_invalid_embedding_vectors_and_quality_values() {
        assert!(SpeakerEmbedding::new(model("v1"), vec![1.0]).is_err());
        assert!(SpeakerEmbedding::new(model("v1"), vec![0.0, 0.0]).is_err());
        assert!(SpeakerEmbedding::new(model("v1"), vec![f32::NAN, 1.0]).is_err());
        assert!(SpeakerEmbedding::new(
            model("v1"),
            vec![1.0; MAX_SPEAKER_EMBEDDING_DIMENSIONS + 1]
        )
        .is_err());
        assert!(SpeakerSampleQuality::new(1, 1.1, 0.8, 0.0).is_err());
        assert!(SpeakerSampleQuality::new(1, 0.8, f32::NAN, 0.0).is_err());
    }
}
