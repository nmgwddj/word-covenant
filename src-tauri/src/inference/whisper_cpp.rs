//! Verified, local-only Whisper adapter backed by `whisper-rs` / whisper.cpp.
//!
//! A caller cannot create this engine from an arbitrary filesystem path. Model
//! loading requires a [`VerifiedModelArtifact`] issued by the native
//! [`ModelRegistry`], after the managed artifact's SHA-256 was rechecked.
//! Neither raw PCM nor filesystem paths are serializable from this module.

use super::{
    asr::{MAX_ASR_EMISSIONS_PER_REQUEST, MAX_TRANSCRIPT_TEXT_BYTES},
    model_registry::{LocalModelKind, RegisteredModel, VerifiedModelArtifact},
    AsrEngine, AsrRequest, AsrResponse, InferenceEngine, InferenceError, ModelProvenance,
    TranscriptEmission, TranscriptEmissionKind,
};
use sha2::{Digest, Sha256};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Exact `input_format` value required on a local model registration.
///
/// `whisper-rs` 0.16 bundles a whisper.cpp loader that checks
/// `GGML_FILE_MAGIC`; it does not accept a GGUF artifact. The filename suffix
/// is not trusted as format evidence.
pub const WHISPER_CPP_GGML_INPUT_FORMAT: &str = "whisper.cpp-ggml";
pub const WHISPER_CPP_PROVIDER: &str = "whisper.cpp";
const MAX_TOKENS_PER_SEGMENT: i32 = 256;
const MAX_INFERENCE_THREADS: usize = 4;
const NANOSECONDS_PER_CENTISECOND: u64 = 10_000_000;

/// Checks the explicit import declaration required by this adapter.
///
/// This is deliberately exact rather than extension- or substring-based, so
/// an imported `gguf` record cannot be mistaken for a compatible GGML model.
pub fn is_whisper_cpp_compatible_input_format(input_format: &str) -> bool {
    input_format == WHISPER_CPP_GGML_INPUT_FORMAT
}

/// Builds the provenance that every Whisper response must carry and validates
/// model metadata before native model loading begins.
pub fn whisper_cpp_model_provenance(
    model: &RegisteredModel,
) -> Result<ModelProvenance, InferenceError> {
    if model.model_kind != LocalModelKind::SpeechRecognition {
        return Err(InferenceError::invalid(
            "the selected local model is not registered for speech recognition",
        ));
    }
    if !is_whisper_cpp_compatible_input_format(&model.input_format) {
        return Err(InferenceError::invalid(format!(
            "the selected local model must declare input format {WHISPER_CPP_GGML_INPUT_FORMAT}"
        )));
    }

    ModelProvenance::new(
        WHISPER_CPP_PROVIDER,
        model.id.to_string(),
        model.version.clone(),
        model.sha256.clone(),
    )
    .map_err(InferenceError::invalid)
}

/// A single-worker local ASR engine. `WhisperState` is mutable and deliberately
/// not shared: the bounded native ASR worker owns one engine instance.
pub struct WhisperCppAsrEngine {
    model_provenance: ModelProvenance,
    state: WhisperState,
    params_template: FullParams<'static, 'static>,
}

impl WhisperCppAsrEngine {
    /// Loads a model from a native-only registry capability.
    ///
    /// The registry checks the managed file's type, size, and digest immediately
    /// before issuing the capability. `whisper-rs` then opens that native path;
    /// the path is dropped after the context is initialized and is never exposed
    /// through a serializable application type.
    pub fn from_registered_artifact(
        artifact: VerifiedModelArtifact,
    ) -> Result<Self, InferenceError> {
        let model_provenance = whisper_cpp_model_provenance(artifact.model())?;

        // Avoid whisper.cpp's default stderr logging, which can expose native
        // model-load details outside the application's audited boundaries.
        whisper_rs::install_logging_hooks();
        let context =
            WhisperContext::new_with_params(artifact.path(), WhisperContextParameters::default())
                .map_err(|_| {
                InferenceError::failed("could not initialize the registered local Whisper model")
            })?;
        if !context.is_multilingual() {
            return Err(InferenceError::invalid(
                "the selected Whisper model must support multilingual transcription",
            ));
        }
        let state = context.create_state().map_err(|_| {
            InferenceError::failed("could not create a local Whisper inference state")
        })?;

        Ok(Self {
            model_provenance,
            state,
            params_template: transcription_params(),
        })
    }
}

impl InferenceEngine for WhisperCppAsrEngine {
    fn model_provenance(&self) -> &ModelProvenance {
        &self.model_provenance
    }
}

impl AsrEngine for WhisperCppAsrEngine {
    fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
        request.validate().map_err(InferenceError::invalid)?;
        whisper_language(request.language.as_deref())?;
        let params = self.params_template.clone();

        self.state
            .full(params, request.audio.samples())
            .map_err(|_| InferenceError::failed("local Whisper transcription failed"))?;

        let native_segment_count = usize::try_from(self.state.full_n_segments()).map_err(|_| {
            InferenceError::failed("local Whisper returned an invalid segment count")
        })?;
        if native_segment_count > MAX_ASR_EMISSIONS_PER_REQUEST {
            return Err(InferenceError::failed(format!(
                "local Whisper produced more than {MAX_ASR_EMISSIONS_PER_REQUEST} final segments for one bounded request"
            )));
        }

        let mut segments = Vec::new();
        for segment in self.state.as_iter() {
            let text = segment
                .to_str()
                .map_err(|_| {
                    InferenceError::failed("local Whisper returned invalid transcript text")
                })?
                .trim();
            if text.is_empty() {
                continue;
            }
            if text.len() > MAX_TRANSCRIPT_TEXT_BYTES {
                return Err(InferenceError::failed(format!(
                    "local Whisper segment exceeds the {MAX_TRANSCRIPT_TEXT_BYTES}-byte transcript limit"
                )));
            }
            if segments.len() >= MAX_ASR_EMISSIONS_PER_REQUEST {
                return Err(InferenceError::failed(format!(
                    "local Whisper produced more than {MAX_ASR_EMISSIONS_PER_REQUEST} final segments for one bounded request"
                )));
            }

            segments.push(WhisperSegmentOutput {
                start_centiseconds: segment.start_timestamp(),
                end_centiseconds: segment.end_timestamp(),
                text: text.to_owned(),
            });
        }

        response_from_whisper_segments(request, &self.model_provenance, segments)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WhisperSegmentOutput {
    start_centiseconds: i64,
    end_centiseconds: i64,
    text: String,
}

fn whisper_language(language: Option<&str>) -> Result<&str, InferenceError> {
    match language {
        None | Some("zh" | "zh-CN" | "zh-Hans" | "ZH" | "ZH-CN") => Ok("zh"),
        Some("auto") => Err(InferenceError::invalid(
            "local Whisper transcription requires an explicit language instead of auto-detection",
        )),
        Some(_) => Err(InferenceError::invalid(
            "the first local Whisper profile supports only the explicit zh language setting",
        )),
    }
}

fn transcription_params() -> FullParams<'static, 'static> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(MAX_INFERENCE_THREADS))
        .unwrap_or(1);
    params.set_n_threads(i32::try_from(threads).expect("bounded inference thread count fits i32"));
    // `whisper-rs` 0.16 keeps its language CString in the parameter object.
    // Store one zh template on the engine and clone it for each request rather
    // than allocating a new language CString for every completed utterance.
    params.set_language(Some("zh"));
    params.set_detect_language(false);
    params.set_translate(false);
    params.set_no_context(true);
    params.set_no_timestamps(false);
    params.set_single_segment(false);
    params.set_token_timestamps(false);
    params.set_max_tokens(MAX_TOKENS_PER_SEGMENT);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_tdrz_enable(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params
}

fn response_from_whisper_segments(
    request: &AsrRequest,
    model_provenance: &ModelProvenance,
    segments: impl IntoIterator<Item = WhisperSegmentOutput>,
) -> Result<AsrResponse, InferenceError> {
    let mut emissions = Vec::new();
    for (index, segment) in segments.into_iter().enumerate() {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }
        if text.len() > MAX_TRANSCRIPT_TEXT_BYTES {
            return Err(InferenceError::failed(format!(
                "local Whisper segment exceeds the {MAX_TRANSCRIPT_TEXT_BYTES}-byte transcript limit"
            )));
        }
        if emissions.len() >= MAX_ASR_EMISSIONS_PER_REQUEST {
            return Err(InferenceError::failed(format!(
                "local Whisper produced more than {MAX_ASR_EMISSIONS_PER_REQUEST} final segments for one bounded request"
            )));
        }

        let (capture_start_ns, capture_end_ns) = capture_range_for_whisper_segment(
            request,
            segment.start_centiseconds,
            segment.end_centiseconds,
        )?;
        emissions.push(TranscriptEmission {
            utterance_key: whisper_utterance_key(request, index, capture_start_ns, capture_end_ns),
            capture_start_ns,
            capture_end_ns,
            text: text.to_owned(),
            kind: TranscriptEmissionKind::Final,
            revision: 1,
            word_timings: Vec::new(),
            model_provenance: model_provenance.clone(),
        });
    }

    AsrResponse::new(request, model_provenance, emissions).map_err(InferenceError::failed)
}

fn capture_range_for_whisper_segment(
    request: &AsrRequest,
    start_centiseconds: i64,
    end_centiseconds: i64,
) -> Result<(u64, u64), InferenceError> {
    if start_centiseconds < 0 || end_centiseconds < 0 {
        return Err(InferenceError::failed(
            "local Whisper returned a negative segment timestamp",
        ));
    }

    let start_offset_ns = u64::try_from(start_centiseconds)
        .ok()
        .and_then(|offset| offset.checked_mul(NANOSECONDS_PER_CENTISECOND))
        .ok_or_else(|| InferenceError::failed("local Whisper segment timestamp overflowed"))?;
    let end_offset_ns = u64::try_from(end_centiseconds)
        .ok()
        .and_then(|offset| offset.checked_mul(NANOSECONDS_PER_CENTISECOND))
        .ok_or_else(|| InferenceError::failed("local Whisper segment timestamp overflowed"))?;
    let duration_ns = request.audio.duration_ns();
    let capture_start_ns = request
        .audio
        .capture_start_ns()
        .checked_add(start_offset_ns.min(duration_ns))
        .ok_or_else(|| InferenceError::failed("local Whisper capture timestamp overflowed"))?;
    let capture_end_ns = request
        .audio
        .capture_start_ns()
        .checked_add(end_offset_ns.min(duration_ns))
        .ok_or_else(|| InferenceError::failed("local Whisper capture timestamp overflowed"))?;
    if capture_end_ns <= capture_start_ns {
        return Err(InferenceError::failed(
            "local Whisper segment does not occupy a valid capture range",
        ));
    }

    Ok((capture_start_ns, capture_end_ns))
}

fn whisper_utterance_key(
    request: &AsrRequest,
    segment_index: usize,
    capture_start_ns: u64,
    capture_end_ns: u64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"word-covenant/whisper-utterance/v1\0");
    digest.update(request.audio.session_id().as_bytes());
    digest.update(capture_start_ns.to_le_bytes());
    digest.update(capture_end_ns.to_le_bytes());
    digest.update(segment_index.to_le_bytes());
    format!("whisper-{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceAudioWindow, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ};
    use chrono::Utc;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn request() -> AsrRequest {
        AsrRequest::new(
            InferenceAudioWindow::new(
                Uuid::nil(),
                1_000,
                1_000_001_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.1; INFERENCE_SAMPLE_RATE_HZ as usize],
            )
            .unwrap(),
            Some("zh".to_owned()),
            true,
        )
        .unwrap()
    }

    fn registered_model(kind: LocalModelKind, input_format: &str) -> RegisteredModel {
        RegisteredModel {
            id: Uuid::nil(),
            model_kind: kind,
            file_path: PathBuf::from("models/registered.model"),
            file_size_bytes: 1,
            sha256: "a".repeat(64),
            version: "v1".to_owned(),
            input_format: input_format.to_owned(),
            model_card_id: "fixture-card".to_owned(),
            license_id: "fixture-license".to_owned(),
            license_confirmed_at: Utc::now(),
            imported_at: Utc::now(),
        }
    }

    #[test]
    fn accepts_only_the_explicit_ggml_registration_format() {
        assert!(is_whisper_cpp_compatible_input_format(
            WHISPER_CPP_GGML_INPUT_FORMAT
        ));
        assert!(!is_whisper_cpp_compatible_input_format("ggml"));
        assert!(!is_whisper_cpp_compatible_input_format("gguf"));
        assert!(!is_whisper_cpp_compatible_input_format("whisper.cpp-ggml "));
    }

    #[test]
    fn rejects_non_asr_or_incompatible_registered_models_before_loading() {
        let wrong_kind = registered_model(
            LocalModelKind::VoiceActivityDetection,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let embedding = registered_model(
            LocalModelKind::SpeakerEmbedding,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let wrong_format = registered_model(LocalModelKind::SpeechRecognition, "gguf");

        assert!(whisper_cpp_model_provenance(&wrong_kind)
            .unwrap_err()
            .to_string()
            .contains("speech recognition"));
        assert!(whisper_cpp_model_provenance(&embedding)
            .unwrap_err()
            .to_string()
            .contains("speech recognition"));
        assert!(whisper_cpp_model_provenance(&wrong_format)
            .unwrap_err()
            .to_string()
            .contains(WHISPER_CPP_GGML_INPUT_FORMAT));
    }

    #[test]
    fn maps_final_whisper_segments_into_validated_audited_provenance() {
        let request = request();
        let model = registered_model(
            LocalModelKind::SpeechRecognition,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let provenance = whisper_cpp_model_provenance(&model).unwrap();

        let response = response_from_whisper_segments(
            &request,
            &provenance,
            [
                WhisperSegmentOutput {
                    start_centiseconds: 0,
                    end_centiseconds: 40,
                    text: "  本机记录  ".to_owned(),
                },
                WhisperSegmentOutput {
                    start_centiseconds: 40,
                    end_centiseconds: 100,
                    text: "无需出网。".to_owned(),
                },
            ],
        )
        .unwrap();

        assert_eq!(response.emissions.len(), 2);
        assert!(response
            .emissions
            .iter()
            .all(|emission| emission.kind == TranscriptEmissionKind::Final));
        assert!(response
            .emissions
            .iter()
            .all(|emission| emission.revision == 1));
        assert!(response
            .emissions
            .iter()
            .all(|emission| emission.model_provenance == provenance));
        assert_eq!(response.emissions[0].capture_start_ns, 1_000);
        assert_eq!(response.emissions[0].capture_end_ns, 400_001_000);
        assert_eq!(response.emissions[0].text, "本机记录");
        assert_eq!(response.emissions[1].capture_start_ns, 400_001_000);
        assert_eq!(response.emissions[1].capture_end_ns, 1_000_001_000);
        assert_ne!(
            response.emissions[0].utterance_key,
            response.emissions[1].utterance_key
        );

        let serialized = serde_json::to_string(&response).unwrap();
        assert!(!serialized.contains("samples"));
    }

    #[test]
    fn accepts_a_no_speech_result_without_synthetic_text() {
        let request = request();
        let model = registered_model(
            LocalModelKind::SpeechRecognition,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let provenance = whisper_cpp_model_provenance(&model).unwrap();

        let response = response_from_whisper_segments(&request, &provenance, []).unwrap();

        assert!(response.emissions.is_empty());
    }

    #[test]
    fn rejects_invalid_or_out_of_window_native_segment_timing() {
        let request = request();
        let model = registered_model(
            LocalModelKind::SpeechRecognition,
            WHISPER_CPP_GGML_INPUT_FORMAT,
        );
        let provenance = whisper_cpp_model_provenance(&model).unwrap();

        let negative = response_from_whisper_segments(
            &request,
            &provenance,
            [WhisperSegmentOutput {
                start_centiseconds: -1,
                end_centiseconds: 10,
                text: "错误".to_owned(),
            }],
        )
        .unwrap_err();
        let out_of_window = response_from_whisper_segments(
            &request,
            &provenance,
            [WhisperSegmentOutput {
                start_centiseconds: 110,
                end_centiseconds: 120,
                text: "错误".to_owned(),
            }],
        )
        .unwrap_err();

        assert!(negative.to_string().contains("negative"));
        assert!(out_of_window.to_string().contains("valid capture range"));
    }

    #[test]
    fn normalizes_chinese_language_tags_and_rejects_auto_detection() {
        assert_eq!(whisper_language(None).unwrap(), "zh");
        assert_eq!(whisper_language(Some("zh-CN")).unwrap(), "zh");
        assert!(whisper_language(Some("auto")).is_err());
        assert!(whisper_language(Some("not-a-language")).is_err());
    }
}
