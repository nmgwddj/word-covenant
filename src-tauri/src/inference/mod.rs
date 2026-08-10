//! Local, bounded contracts for speech inference.
//!
//! This module intentionally contains no endpoint, download, or credential
//! configuration. Audio is prepared in Rust and stays in-process; callers must
//! not serialize a PCM window across the WebView boundary.

pub mod asr;
pub(crate) mod bundled_model;
pub mod gap;
pub mod mock;
pub mod model_registry;
pub mod pipeline;
pub mod vad;
pub mod webrtc_vad;
pub mod whisper_cpp;

pub use asr::{
    AsrEngine, AsrFinalIdempotencyKey, AsrRequest, AsrResponse, FinalTranscriptEmission,
    MappedTranscriptEmission, TranscriptEmission, TranscriptEmissionKind, TranscriptEmissionMapper,
    TranscriptWordTiming, TransientTranscriptEmission,
};
pub use gap::{InferenceGap, InferenceGapReason, InferenceGapStage};
pub use mock::{FixtureAsr, FixtureAsrCue, FixtureVad, FixtureVadCue};
pub use vad::{VadEngine, VadRequest, VadResponse, VoiceActivitySegment};
pub use webrtc_vad::{
    WebRtcVad, WebRtcVadMode, WEBRTC_VAD_FRAME_DURATION_MS, WEBRTC_VAD_FRAME_SAMPLES,
};
pub use whisper_cpp::{
    is_whisper_cpp_compatible_input_format, WhisperCppAsrEngine, WHISPER_CPP_GGML_INPUT_FORMAT,
};

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// The only PCM format accepted by the M2 inference boundary.
///
/// Capture code may use another device format, but must resample and downmix
/// before constructing an [`InferenceAudioWindow`].
pub const INFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;
pub const INFERENCE_CHANNELS: u16 = 1;
pub const MAX_INFERENCE_WINDOW_SAMPLES: usize = 480_000;
pub const MAX_INFERENCE_WINDOW_DURATION_NS: u64 = 30_000_000_000;
pub const MAX_MODEL_IDENTIFIER_BYTES: usize = 128;
pub const SHA256_HEX_LENGTH: usize = 64;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// A bounded, normalized PCM window that remains inside the native process.
///
/// `capture_*_ns` use the session's monotonic capture clock. The timestamp
/// range deliberately survives resampling so inference emissions can retain
/// their original capture times.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceAudioWindow {
    session_id: Uuid,
    capture_start_ns: u64,
    capture_end_ns: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: Vec<f32>,
}

impl InferenceAudioWindow {
    pub fn new(
        session_id: Uuid,
        capture_start_ns: u64,
        capture_end_ns: u64,
        sample_rate_hz: u32,
        channels: u16,
        samples: Vec<f32>,
    ) -> Result<Self, String> {
        let window = Self {
            session_id,
            capture_start_ns,
            capture_end_ns,
            sample_rate_hz,
            channels,
            samples,
        };
        window.validate()?;
        Ok(window)
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn capture_start_ns(&self) -> u64 {
        self.capture_start_ns
    }

    pub fn capture_end_ns(&self) -> u64 {
        self.capture_end_ns
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    pub fn duration_ns(&self) -> u64 {
        self.capture_end_ns.saturating_sub(self.capture_start_ns)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate_hz != INFERENCE_SAMPLE_RATE_HZ {
            return Err(format!(
                "inference audio sample rate must be {INFERENCE_SAMPLE_RATE_HZ} Hz"
            ));
        }
        if self.channels != INFERENCE_CHANNELS {
            return Err(format!(
                "inference audio channel count must be {INFERENCE_CHANNELS}"
            ));
        }
        if self.samples.is_empty() {
            return Err("inference audio window must contain samples".to_owned());
        }
        if self.samples.len() > MAX_INFERENCE_WINDOW_SAMPLES {
            return Err(format!(
                "inference audio window exceeds {MAX_INFERENCE_WINDOW_SAMPLES} samples"
            ));
        }
        if !self.samples.iter().all(|sample| sample.is_finite()) {
            return Err("inference audio samples must be finite".to_owned());
        }
        if self.capture_end_ns < self.capture_start_ns {
            return Err("inference audio end must not precede its start".to_owned());
        }

        let expected_duration_ns = duration_for_frames(self.frame_count());
        let actual_duration_ns = self.duration_ns();
        let one_frame_ns = duration_for_frames(1);
        if actual_duration_ns.abs_diff(expected_duration_ns) > one_frame_ns {
            return Err("inference audio timestamps do not match its PCM duration".to_owned());
        }
        if actual_duration_ns > MAX_INFERENCE_WINDOW_DURATION_NS {
            return Err(format!(
                "inference audio window exceeds {MAX_INFERENCE_WINDOW_DURATION_NS} ns"
            ));
        }

        Ok(())
    }
}

/// Model and runtime identity recorded with every inference output.
///
/// The artifact digest identifies bytes that the user imported locally. This
/// type records provenance; integrity verification remains the registry's job.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", try_from = "ModelProvenanceWire")]
pub struct ModelProvenance {
    provider: String,
    model_id: String,
    model_version: String,
    artifact_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelProvenanceWire {
    provider: String,
    model_id: String,
    model_version: String,
    artifact_sha256: String,
}

impl ModelProvenance {
    pub fn new(
        provider: impl Into<String>,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        artifact_sha256: impl Into<String>,
    ) -> Result<Self, String> {
        let provenance = Self {
            provider: provider.into(),
            model_id: model_id.into(),
            model_version: model_version.into(),
            artifact_sha256: artifact_sha256.into().to_ascii_lowercase(),
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_identifier("model provider", &self.provider)?;
        validate_identifier("model id", &self.model_id)?;
        validate_identifier("model version", &self.model_version)?;

        if self.artifact_sha256.len() != SHA256_HEX_LENGTH
            || !self
                .artifact_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("model artifact hash must be a SHA-256 hex digest".to_owned());
        }

        Ok(())
    }
}

impl TryFrom<ModelProvenanceWire> for ModelProvenance {
    type Error = String;

    fn try_from(value: ModelProvenanceWire) -> Result<Self, Self::Error> {
        Self::new(
            value.provider,
            value.model_id,
            value.model_version,
            value.artifact_sha256,
        )
    }
}

/// This contract has no remote execution variant by design.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceExecutionScope {
    OnDevice,
}

/// Shared capability of an in-process VAD or ASR adapter.
///
/// Implementations must execute locally and never fetch a model or send audio,
/// transcripts, or model metadata over the network. Egress remains outside
/// this trait and requires the application policy gate.
pub trait InferenceEngine: Send {
    fn model_provenance(&self) -> &ModelProvenance;

    fn execution_scope(&self) -> InferenceExecutionScope {
        InferenceExecutionScope::OnDevice
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "message")]
pub enum InferenceError {
    InvalidInput(String),
    BackendUnavailable(String),
    Failed(String),
}

impl InferenceError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidInput(bound_error_message(message.into()))
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::BackendUnavailable(bound_error_message(message.into()))
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(bound_error_message(message.into()))
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::BackendUnavailable(message)
            | Self::Failed(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for InferenceError {}

pub(crate) fn duration_for_frames(frame_count: usize) -> u64 {
    ((frame_count as u128)
        .saturating_mul(NANOS_PER_SECOND)
        .checked_div(u128::from(INFERENCE_SAMPLE_RATE_HZ))
        .unwrap_or(u128::from(u64::MAX)))
    .min(u128::from(u64::MAX)) as u64
}

pub(crate) fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    if value.len() > MAX_MODEL_IDENTIFIER_BYTES {
        return Err(format!(
            "{label} exceeds {MAX_MODEL_IDENTIFIER_BYTES} bytes"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{label} must not contain control characters"));
    }

    Ok(())
}

fn bound_error_message(mut message: String) -> String {
    const MAX_ERROR_MESSAGE_BYTES: usize = 512;

    if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        return message;
    }

    let mut end = MAX_ERROR_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_window() -> InferenceAudioWindow {
        InferenceAudioWindow::new(
            Uuid::nil(),
            1_000,
            1_000_001_000,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; INFERENCE_SAMPLE_RATE_HZ as usize],
        )
        .unwrap()
    }

    #[test]
    fn accepts_a_bounded_normalized_audio_window() {
        let window = audio_window();

        assert_eq!(window.frame_count(), 16_000);
        assert_eq!(window.duration_ns(), 1_000_000_000);
        assert_eq!(window.sample_rate_hz(), INFERENCE_SAMPLE_RATE_HZ);
    }

    #[test]
    fn rejects_non_normalized_or_misaligned_audio() {
        assert!(InferenceAudioWindow::new(
            Uuid::nil(),
            0,
            1_000_000_000,
            48_000,
            INFERENCE_CHANNELS,
            vec![0.0; 16_000],
        )
        .is_err());
        assert!(InferenceAudioWindow::new(
            Uuid::nil(),
            0,
            999_000_000,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; 16_000],
        )
        .is_err());
    }

    #[test]
    fn validates_model_provenance_during_deserialization() {
        let serialized = serde_json::json!({
            "provider": "fixture",
            "modelId": "fixture-asr",
            "modelVersion": "v1",
            "artifactSha256": "AaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaA"
        });

        let provenance: ModelProvenance = serde_json::from_value(serialized).unwrap();

        assert_eq!(
            provenance.artifact_sha256(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
    }
}
