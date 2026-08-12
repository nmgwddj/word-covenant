use super::{SpeakerEmbedding, SpeakerEmbeddingEngine, SpeakerMatchPolicy, SpeakerSampleQuality};
use crate::inference::{
    InferenceAudioWindow, InferenceEngine, InferenceError, InferenceExecutionScope,
    ModelProvenance, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ,
};
use serde::Deserialize;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::fs;
use std::path::{Component, Path, PathBuf};

const MANIFEST_RELATIVE_PATH: &str = "models/speaker-embedding/manifest.json";
const EMBEDDED_MANIFEST: &str =
    include_str!("../../resources/models/speaker-embedding/manifest.json");

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerModelManifest {
    schema_version: u32,
    provider: String,
    model_id: String,
    model_version: String,
    artifact_file_name: String,
    artifact_sha256: String,
    size_bytes: u64,
    sample_rate_hz: u32,
    embedding_dimensions: usize,
    runtime: String,
    runtime_version: String,
    license_id: String,
    model_card_id: String,
    threshold_status: String,
    minimum_similarity: f32,
    minimum_runner_up_margin: f32,
    anonymous_minimum_similarity: f32,
    anonymous_minimum_runner_up_margin: f32,
    minimum_voiced_duration_ns: u64,
    minimum_voiced_ratio: f32,
    minimum_signal_quality: f32,
    maximum_overlap_probability: f32,
    ready_confirmed_duration_ns: u64,
}

impl SpeakerModelManifest {
    pub fn provenance(&self) -> Result<ModelProvenance, String> {
        ModelProvenance::new(
            self.provider.clone(),
            self.model_id.clone(),
            self.model_version.clone(),
            self.artifact_sha256.clone(),
        )
    }

    pub fn profile_match_policy(&self) -> Result<SpeakerMatchPolicy, String> {
        SpeakerMatchPolicy::new(
            self.minimum_similarity,
            self.minimum_runner_up_margin,
            self.minimum_voiced_duration_ns,
            self.minimum_voiced_ratio,
            self.minimum_signal_quality,
            self.maximum_overlap_probability,
        )
    }

    pub fn anonymous_match_policy(&self) -> Result<SpeakerMatchPolicy, String> {
        SpeakerMatchPolicy::new(
            self.anonymous_minimum_similarity,
            self.anonymous_minimum_runner_up_margin,
            self.minimum_voiced_duration_ns,
            self.minimum_voiced_ratio,
            self.minimum_signal_quality,
            self.maximum_overlap_probability,
        )
    }

    pub fn embedding_dimensions(&self) -> usize {
        self.embedding_dimensions
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.artifact_file_name != "model.onnx"
            || self.size_bytes == 0
            || self.sample_rate_hz != INFERENCE_SAMPLE_RATE_HZ
            || self.embedding_dimensions != 192
            || self.runtime != "sherpa-onnx"
            || self.runtime_version.trim().is_empty()
            || self.license_id != "Apache-2.0"
            || self.model_card_id.trim().is_empty()
            || self.threshold_status != "provisional-pending-consented-corpus-calibration"
            || self.ready_confirmed_duration_ns != 4_000_000_000
        {
            return Err("bundled speaker model manifest has invalid metadata".to_owned());
        }
        self.provenance()?;
        self.profile_match_policy()?;
        self.anonymous_match_policy()?;
        Ok(())
    }
}

pub struct BundledSpeakerModel {
    pub manifest: SpeakerModelManifest,
    pub artifact_path: PathBuf,
}

pub fn bundled_speaker_model(resource_dir: &Path) -> Result<BundledSpeakerModel, String> {
    let resource_root = canonical_resource_root(resource_dir)?;
    let expected: SpeakerModelManifest = serde_json::from_str(EMBEDDED_MANIFEST)
        .map_err(|_| "embedded speaker model manifest is invalid".to_owned())?;
    expected.validate()?;
    let manifest_path = resolve_regular_file(&resource_root, Path::new(MANIFEST_RELATIVE_PATH))?;
    let packaged: SpeakerModelManifest = serde_json::from_str(
        &fs::read_to_string(manifest_path)
            .map_err(|_| "bundled speaker model manifest is unreadable".to_owned())?,
    )
    .map_err(|_| "bundled speaker model manifest is invalid".to_owned())?;
    if packaged != expected {
        return Err("bundled speaker model manifest does not match the application".to_owned());
    }
    let relative_artifact =
        Path::new("models/speaker-embedding").join(&expected.artifact_file_name);
    let artifact_path = resolve_regular_file(&resource_root, &relative_artifact)?;
    let metadata = fs::metadata(&artifact_path)
        .map_err(|_| "bundled speaker model cannot be inspected".to_owned())?;
    if metadata.len() != expected.size_bytes {
        return Err("bundled speaker model has an unexpected size".to_owned());
    }
    Ok(BundledSpeakerModel {
        manifest: expected,
        artifact_path,
    })
}

pub struct OnnxSpeakerEmbeddingEngine {
    provenance: ModelProvenance,
    manifest: SpeakerModelManifest,
    extractor: EmbeddingExtractor,
}

impl OnnxSpeakerEmbeddingEngine {
    pub fn from_bundled(model: BundledSpeakerModel) -> Result<Self, String> {
        let provenance = model.manifest.provenance()?;
        let extractor = EmbeddingExtractor::new(ExtractorConfig {
            model: model.artifact_path.to_string_lossy().into_owned(),
            provider: Some("cpu".to_owned()),
            num_threads: Some(2),
            debug: false,
        })
        .map_err(|_| "bundled speaker model could not be loaded".to_owned())?;
        if extractor.embedding_size != model.manifest.embedding_dimensions {
            return Err("bundled speaker model returned an unexpected embedding size".to_owned());
        }
        Ok(Self {
            provenance,
            manifest: model.manifest,
            extractor,
        })
    }

    pub fn manifest(&self) -> &SpeakerModelManifest {
        &self.manifest
    }
}

impl InferenceEngine for OnnxSpeakerEmbeddingEngine {
    fn model_provenance(&self) -> &ModelProvenance {
        &self.provenance
    }

    fn execution_scope(&self) -> InferenceExecutionScope {
        InferenceExecutionScope::OnDevice
    }
}

impl SpeakerEmbeddingEngine for OnnxSpeakerEmbeddingEngine {
    fn embed(
        &mut self,
        audio: &InferenceAudioWindow,
    ) -> Result<(SpeakerEmbedding, SpeakerSampleQuality), InferenceError> {
        audio.validate().map_err(InferenceError::invalid)?;
        if audio.sample_rate_hz() != INFERENCE_SAMPLE_RATE_HZ
            || audio.channels() != INFERENCE_CHANNELS
        {
            return Err(InferenceError::invalid(
                "speaker embedding requires 16 kHz mono PCM",
            ));
        }
        let values = self
            .extractor
            .compute_speaker_embedding(audio.samples().to_vec(), audio.sample_rate_hz())
            .map_err(|_| InferenceError::failed("local speaker embedding inference failed"))?;
        let embedding = SpeakerEmbedding::new(self.provenance.clone(), values)
            .map_err(InferenceError::failed)?;
        if embedding.dimensions() != self.manifest.embedding_dimensions {
            return Err(InferenceError::failed(
                "speaker embedding dimensions do not match the bundled manifest",
            ));
        }

        let (voiced_ratio, signal_quality) = waveform_quality(audio.samples());
        let quality =
            SpeakerSampleQuality::new(audio.duration_ns(), voiced_ratio, signal_quality, 0.0)
                .map_err(InferenceError::failed)?;
        Ok((embedding, quality))
    }
}

fn waveform_quality(samples: &[f32]) -> (f32, f32) {
    const FRAME_SAMPLES: usize = 320;
    const VOICED_RMS: f64 = 0.008;
    const MINIMUM_USEFUL_DBFS: f64 = -42.0;
    const STRONG_SPEECH_DBFS: f64 = -24.0;
    let mut frame_count = 0_usize;
    let mut voiced_frames = 0_usize;
    let mut voiced_rms_sum = 0.0_f64;
    for frame in samples.chunks(FRAME_SAMPLES) {
        let rms = (frame
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / frame.len() as f64)
            .sqrt();
        frame_count += 1;
        if rms >= VOICED_RMS {
            voiced_frames += 1;
            voiced_rms_sum += rms;
        }
    }
    if frame_count == 0 {
        return (0.0, 0.0);
    }
    let voiced_ratio = voiced_frames as f32 / frame_count as f32;
    if voiced_frames == 0 {
        return (voiced_ratio, 0.0);
    }
    // Pre-roll and pauses are expected in VAD windows, so loudness is measured
    // only across voiced frames. Normalize the result in dBFS: ordinary speech
    // around -32 dBFS clears the provisional quality gate, while barely audible
    // frames near the VAD floor remain ineligible for biometric enrollment.
    let average_voiced_rms = voiced_rms_sum / voiced_frames as f64;
    let voiced_dbfs = 20.0 * average_voiced_rms.log10();
    let signal_quality = ((voiced_dbfs - MINIMUM_USEFUL_DBFS)
        / (STRONG_SPEECH_DBFS - MINIMUM_USEFUL_DBFS))
        .clamp(0.0, 1.0) as f32;
    (voiced_ratio, signal_quality)
}

fn canonical_resource_root(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("speaker resource directory must be absolute".to_owned());
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| "speaker resource directory is unavailable".to_owned())?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err("speaker resource directory is invalid".to_owned());
    }
    fs::canonicalize(path).map_err(|_| "speaker resource directory is unavailable".to_owned())
}

fn resolve_regular_file(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.is_absolute()
        || relative.components().next().is_none()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err("speaker resource path is invalid".to_owned());
    }
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            unreachable!();
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|_| "bundled speaker resource is missing".to_owned())?;
        if metadata.file_type().is_symlink() {
            return Err("bundled speaker resource must not be a symlink".to_owned());
        }
    }
    let metadata =
        fs::metadata(&current).map_err(|_| "bundled speaker resource is unavailable".to_owned())?;
    if !metadata.is_file() {
        return Err("bundled speaker resource is not a regular file".to_owned());
    }
    let canonical = fs::canonicalize(current)
        .map_err(|_| "bundled speaker resource is unavailable".to_owned())?;
    if !canonical.starts_with(root) {
        return Err("bundled speaker resource escaped its resource directory".to_owned());
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_defines_valid_model_spaces_and_policies() {
        let manifest: SpeakerModelManifest = serde_json::from_str(EMBEDDED_MANIFEST).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.embedding_dimensions(), 192);
        assert_ne!(
            manifest.profile_match_policy().unwrap(),
            manifest.anonymous_match_policy().unwrap()
        );
    }

    #[test]
    fn waveform_quality_is_bounded_and_rejects_silence() {
        assert_eq!(waveform_quality(&vec![0.0; 16_000]), (0.0, 0.0));
        let (ratio, signal) = waveform_quality(&vec![0.25; 16_000]);
        assert_eq!(ratio, 1.0);
        assert_eq!(signal, 1.0);
    }

    #[test]
    fn realistic_microphone_speech_passes_the_provisional_quality_policy() {
        let mut samples = Vec::with_capacity(19_200);
        for frame in 0..60 {
            let amplitude = if frame < 45 { 0.04 } else { 0.0 };
            samples.extend((0..320).map(|sample| {
                if sample % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                }
            }));
        }
        let (voiced_ratio, signal_quality) = waveform_quality(&samples);
        let quality =
            SpeakerSampleQuality::new(1_200_000_000, voiced_ratio, signal_quality, 0.0).unwrap();
        let manifest: SpeakerModelManifest = serde_json::from_str(EMBEDDED_MANIFEST).unwrap();

        assert_eq!(voiced_ratio, 0.75);
        assert!(signal_quality >= 0.45);
        assert_eq!(
            manifest
                .profile_match_policy()
                .unwrap()
                .sample_rejection(quality),
            None
        );
    }

    #[test]
    fn barely_audible_voiced_frames_remain_ineligible() {
        let samples = (0..16_000)
            .map(|sample| if sample % 2 == 0 { 0.009 } else { -0.009 })
            .collect::<Vec<_>>();
        let (voiced_ratio, signal_quality) = waveform_quality(&samples);
        let quality =
            SpeakerSampleQuality::new(1_000_000_000, voiced_ratio, signal_quality, 0.0).unwrap();
        let manifest: SpeakerModelManifest = serde_json::from_str(EMBEDDED_MANIFEST).unwrap();

        assert_eq!(voiced_ratio, 1.0);
        assert!(signal_quality < 0.45);
        assert_eq!(
            manifest
                .profile_match_policy()
                .unwrap()
                .sample_rejection(quality),
            Some(super::super::SpeakerSampleRejection::LowSignalQuality)
        );
    }

    #[test]
    #[ignore = "requires the optional local speaker embedding model"]
    fn packaged_model_loads_with_the_expected_embedding_dimension() {
        let resource_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources");
        let model = bundled_speaker_model(&resource_dir).unwrap();
        let engine = OnnxSpeakerEmbeddingEngine::from_bundled(model).unwrap();

        assert_eq!(engine.extractor.embedding_size, 192);
        assert_eq!(engine.manifest().embedding_dimensions(), 192);
    }
}
