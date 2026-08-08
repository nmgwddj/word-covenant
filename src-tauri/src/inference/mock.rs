//! Deterministic in-process fixtures for inference integration tests.
//!
//! These adapters contain fixed metadata only. They do not open model files,
//! start a runtime, fetch assets, or make network requests.

use super::asr::{
    AsrEngine, AsrRequest, AsrResponse, TranscriptEmission, TranscriptEmissionKind,
    MAX_ASR_EMISSIONS_PER_REQUEST,
};
use super::vad::{
    VadEngine, VadRequest, VadResponse, VoiceActivitySegment, MAX_VAD_SEGMENTS_PER_REQUEST,
};
use super::{InferenceEngine, InferenceError, ModelProvenance, MAX_INFERENCE_WINDOW_DURATION_NS};
use serde::{Deserialize, Serialize};

const FIXTURE_ARTIFACT_SHA256: &str =
    "f3a6e1ab2873d47c5f3299d42110f6b9e5c8472d3a91bc6d0482f4e7a50619cd";
const FIXTURE_SPEECH_DURATION_NS: u64 = 1_000_000_000;
const FIXTURE_PARTIAL_END_NS: u64 = 500_000_000;

/// A VAD cue relative to an inference request's capture start time.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FixtureVadCue {
    pub start_offset_ns: u64,
    pub end_offset_ns: u64,
    pub speech_probability: f32,
}

impl FixtureVadCue {
    fn validate(&self) -> Result<(), String> {
        VoiceActivitySegment {
            capture_start_ns: self.start_offset_ns,
            capture_end_ns: self.end_offset_ns,
            speech_probability: self.speech_probability,
        }
        .validate()?;
        if self.end_offset_ns > MAX_INFERENCE_WINDOW_DURATION_NS {
            return Err("fixture VAD cue exceeds the maximum inference window".to_owned());
        }
        Ok(())
    }
}

/// A transcript cue relative to an inference request's capture start time.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FixtureAsrCue {
    pub utterance_key: String,
    pub start_offset_ns: u64,
    pub end_offset_ns: u64,
    pub text: String,
    pub kind: TranscriptEmissionKind,
    pub revision: u32,
}

impl FixtureAsrCue {
    fn validate(&self, model_provenance: ModelProvenance) -> Result<(), String> {
        TranscriptEmission {
            utterance_key: self.utterance_key.clone(),
            capture_start_ns: self.start_offset_ns,
            capture_end_ns: self.end_offset_ns,
            text: self.text.clone(),
            kind: self.kind,
            revision: self.revision,
            word_timings: Vec::new(),
            model_provenance,
        }
        .validate()?;
        if self.end_offset_ns > MAX_INFERENCE_WINDOW_DURATION_NS {
            return Err("fixture ASR cue exceeds the maximum inference window".to_owned());
        }
        Ok(())
    }
}

/// A deterministic VAD fixture that maps a static cue sheet onto the supplied
/// capture timestamps.
pub struct FixtureVad {
    model_provenance: ModelProvenance,
    cues: Vec<FixtureVadCue>,
}

impl FixtureVad {
    pub fn new(
        model_provenance: ModelProvenance,
        cues: Vec<FixtureVadCue>,
    ) -> Result<Self, String> {
        model_provenance.validate()?;
        if cues.len() > MAX_VAD_SEGMENTS_PER_REQUEST {
            return Err(format!(
                "fixture VAD exceeds {MAX_VAD_SEGMENTS_PER_REQUEST} cues"
            ));
        }
        for cue in &cues {
            cue.validate()?;
        }

        Ok(Self {
            model_provenance,
            cues,
        })
    }

    pub fn scripted() -> Self {
        Self::new(
            fixture_provenance("fixture-vad"),
            vec![FixtureVadCue {
                start_offset_ns: 0,
                end_offset_ns: FIXTURE_SPEECH_DURATION_NS,
                speech_probability: 0.98,
            }],
        )
        .expect("built-in VAD fixture is valid")
    }
}

impl Default for FixtureVad {
    fn default() -> Self {
        Self::scripted()
    }
}

impl InferenceEngine for FixtureVad {
    fn model_provenance(&self) -> &ModelProvenance {
        &self.model_provenance
    }
}

impl VadEngine for FixtureVad {
    fn detect_voice_activity(
        &mut self,
        request: &VadRequest,
    ) -> Result<VadResponse, InferenceError> {
        let mut segments = Vec::with_capacity(self.cues.len());
        for cue in &self.cues {
            segments.push(VoiceActivitySegment {
                capture_start_ns: apply_offset(
                    request.audio.capture_start_ns(),
                    cue.start_offset_ns,
                )?,
                capture_end_ns: apply_offset(request.audio.capture_start_ns(), cue.end_offset_ns)?,
                speech_probability: cue.speech_probability,
            });
        }

        VadResponse::new(request, self.model_provenance.clone(), segments)
            .map_err(InferenceError::invalid)
    }
}

/// A deterministic ASR fixture that emits a partial revision followed by a
/// final revision for the same utterance key.
pub struct FixtureAsr {
    model_provenance: ModelProvenance,
    cues: Vec<FixtureAsrCue>,
}

impl FixtureAsr {
    pub fn new(
        model_provenance: ModelProvenance,
        cues: Vec<FixtureAsrCue>,
    ) -> Result<Self, String> {
        model_provenance.validate()?;
        if cues.len() > MAX_ASR_EMISSIONS_PER_REQUEST {
            return Err(format!(
                "fixture ASR exceeds {MAX_ASR_EMISSIONS_PER_REQUEST} cues"
            ));
        }
        for cue in &cues {
            cue.validate(model_provenance.clone())?;
        }

        Ok(Self {
            model_provenance,
            cues,
        })
    }

    pub fn scripted() -> Self {
        Self::new(
            fixture_provenance("fixture-asr"),
            vec![
                FixtureAsrCue {
                    utterance_key: "fixture-utterance-1".to_owned(),
                    start_offset_ns: 0,
                    end_offset_ns: FIXTURE_PARTIAL_END_NS,
                    text: "本次记录仅".to_owned(),
                    kind: TranscriptEmissionKind::Partial,
                    revision: 1,
                },
                FixtureAsrCue {
                    utterance_key: "fixture-utterance-1".to_owned(),
                    start_offset_ns: 0,
                    end_offset_ns: FIXTURE_SPEECH_DURATION_NS,
                    text: "本次记录仅保存在本机。".to_owned(),
                    kind: TranscriptEmissionKind::Final,
                    revision: 2,
                },
            ],
        )
        .expect("built-in ASR fixture is valid")
    }
}

impl Default for FixtureAsr {
    fn default() -> Self {
        Self::scripted()
    }
}

impl InferenceEngine for FixtureAsr {
    fn model_provenance(&self) -> &ModelProvenance {
        &self.model_provenance
    }
}

impl AsrEngine for FixtureAsr {
    fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
        let mut emissions = Vec::with_capacity(self.cues.len());
        for cue in &self.cues {
            if !request.emit_partials && cue.kind == TranscriptEmissionKind::Partial {
                continue;
            }
            emissions.push(TranscriptEmission {
                utterance_key: cue.utterance_key.clone(),
                capture_start_ns: apply_offset(
                    request.audio.capture_start_ns(),
                    cue.start_offset_ns,
                )?,
                capture_end_ns: apply_offset(request.audio.capture_start_ns(), cue.end_offset_ns)?,
                text: cue.text.clone(),
                kind: cue.kind,
                revision: cue.revision,
                word_timings: Vec::new(),
                model_provenance: self.model_provenance.clone(),
            });
        }

        AsrResponse::new(request, &self.model_provenance, emissions)
            .map_err(InferenceError::invalid)
    }
}

fn fixture_provenance(model_id: &str) -> ModelProvenance {
    ModelProvenance::new(
        "word-covenant-fixture",
        model_id,
        "fixture-v1",
        FIXTURE_ARTIFACT_SHA256,
    )
    .expect("built-in fixture model provenance is valid")
}

fn apply_offset(capture_start_ns: u64, offset_ns: u64) -> Result<u64, InferenceError> {
    capture_start_ns.checked_add(offset_ns).ok_or_else(|| {
        InferenceError::invalid("fixture capture offset overflows the monotonic clock")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        AsrEngine, InferenceAudioWindow, InferenceExecutionScope, VadEngine, INFERENCE_CHANNELS,
        INFERENCE_SAMPLE_RATE_HZ,
    };
    use uuid::Uuid;

    fn audio_window() -> InferenceAudioWindow {
        InferenceAudioWindow::new(
            Uuid::nil(),
            10_000,
            1_000_010_000,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.0; 16_000],
        )
        .unwrap()
    }

    #[test]
    fn fixture_vad_maps_its_script_to_the_capture_clock() {
        let mut fixture = FixtureVad::default();
        let response = fixture
            .detect_voice_activity(&VadRequest::new(audio_window()).unwrap())
            .unwrap();

        assert_eq!(fixture.execution_scope(), InferenceExecutionScope::OnDevice);
        assert_eq!(response.segments.len(), 1);
        assert_eq!(response.segments[0].capture_start_ns, 10_000);
        assert_eq!(response.segments[0].capture_end_ns, 1_000_010_000);
        assert_eq!(response.model_provenance.model_version(), "fixture-v1");
    }

    #[test]
    fn fixture_asr_emits_stable_partial_and_final_revisions() {
        let mut fixture = FixtureAsr::default();
        let request = AsrRequest::new(audio_window(), Some("zh".to_owned()), true).unwrap();

        let first = fixture.transcribe(&request).unwrap();
        let second = fixture.transcribe(&request).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first
                .emissions
                .iter()
                .map(|emission| (emission.kind, emission.revision))
                .collect::<Vec<_>>(),
            vec![
                (TranscriptEmissionKind::Partial, 1),
                (TranscriptEmissionKind::Final, 2),
            ]
        );
        assert_eq!(
            first.emissions[1].model_provenance.model_id(),
            "fixture-asr"
        );
    }

    #[test]
    fn fixture_omits_partials_when_the_request_disables_them() {
        let mut fixture = FixtureAsr::default();
        let request = AsrRequest::new(audio_window(), Some("zh".to_owned()), false).unwrap();

        let response = fixture.transcribe(&request).unwrap();

        assert_eq!(response.emissions.len(), 1);
        assert_eq!(response.emissions[0].kind, TranscriptEmissionKind::Final);
    }
}
