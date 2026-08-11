use super::{
    InferenceAudioWindow, InferenceEngine, InferenceError, ModelProvenance,
    MAX_MODEL_IDENTIFIER_BYTES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

pub const MAX_ASR_EMISSIONS_PER_REQUEST: usize = 16;
pub const MAX_TRANSCRIPT_TEXT_BYTES: usize = 4_096;
pub const MAX_WORD_TIMINGS_PER_EMISSION: usize = 256;
pub const MAX_LANGUAGE_TAG_BYTES: usize = 35;
pub const MAX_TRANSIENT_ASR_UTTERANCES: usize = 256;

/// Input for one bounded local ASR call.
#[derive(Clone, Debug, PartialEq)]
pub struct AsrRequest {
    pub audio: InferenceAudioWindow,
    pub language: Option<String>,
    pub emit_partials: bool,
}

impl AsrRequest {
    pub fn new(
        audio: InferenceAudioWindow,
        language: Option<String>,
        emit_partials: bool,
    ) -> Result<Self, String> {
        let request = Self {
            audio,
            language,
            emit_partials,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.audio.validate()?;
        if let Some(language) = &self.language {
            if language.trim().is_empty() {
                return Err("ASR language tag must not be empty".to_owned());
            }
            if language.len() > MAX_LANGUAGE_TAG_BYTES {
                return Err(format!(
                    "ASR language tag exceeds {MAX_LANGUAGE_TAG_BYTES} bytes"
                ));
            }
            if language.chars().any(char::is_control) {
                return Err("ASR language tag must not contain control characters".to_owned());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptEmissionKind {
    Partial,
    Final,
}

/// Optional word-level timing inside one transcript emission.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptWordTiming {
    pub text: String,
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
}

/// A revisioned local ASR result. Partial output is display-only; downstream
/// Agent context must select only [`TranscriptEmissionKind::Final`] records.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptEmission {
    /// Stable across partial and final revisions of the same utterance.
    pub utterance_key: String,
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
    pub text: String,
    pub kind: TranscriptEmissionKind,
    pub revision: u32,
    pub word_timings: Vec<TranscriptWordTiming>,
    pub model_provenance: ModelProvenance,
}

impl TranscriptEmission {
    pub fn validate(&self) -> Result<(), String> {
        validate_utterance_key(&self.utterance_key)?;
        if self.capture_end_ns <= self.capture_start_ns {
            return Err("transcript emission end must follow its start".to_owned());
        }
        if self.text.trim().is_empty() {
            return Err("transcript emission text must not be empty".to_owned());
        }
        if self.text.len() > MAX_TRANSCRIPT_TEXT_BYTES {
            return Err(format!(
                "transcript emission text exceeds {MAX_TRANSCRIPT_TEXT_BYTES} bytes"
            ));
        }
        if self.revision == 0 {
            return Err("transcript emission revision must start at 1".to_owned());
        }
        if self.word_timings.len() > MAX_WORD_TIMINGS_PER_EMISSION {
            return Err(format!(
                "transcript emission exceeds {MAX_WORD_TIMINGS_PER_EMISSION} word timings"
            ));
        }
        self.model_provenance.validate()?;

        let mut previous_end = None;
        for timing in &self.word_timings {
            if timing.text.trim().is_empty() {
                return Err("word timing text must not be empty".to_owned());
            }
            if timing.text.len() > MAX_TRANSCRIPT_TEXT_BYTES {
                return Err("word timing text exceeds transcript text limit".to_owned());
            }
            if timing.capture_end_ns <= timing.capture_start_ns {
                return Err("word timing end must follow its start".to_owned());
            }
            if timing.capture_start_ns < self.capture_start_ns
                || timing.capture_end_ns > self.capture_end_ns
            {
                return Err("word timing must remain inside its transcript emission".to_owned());
            }
            if previous_end.is_some_and(|end| timing.capture_start_ns < end) {
                return Err("word timings must be ordered and non-overlapping".to_owned());
            }
            previous_end = Some(timing.capture_end_ns);
        }

        Ok(())
    }
}

/// Declares whether an empty local ASR response is intentional no-speech.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrResponseDisposition {
    #[default]
    Transcript,
    NoSpeech,
}

/// Bounded emissions produced from a single local ASR request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrResponse {
    pub emissions: Vec<TranscriptEmission>,
    #[serde(default)]
    pub disposition: AsrResponseDisposition,
}

impl AsrResponse {
    pub fn new(
        request: &AsrRequest,
        expected_model: &ModelProvenance,
        emissions: Vec<TranscriptEmission>,
    ) -> Result<Self, String> {
        let response = Self {
            emissions,
            disposition: AsrResponseDisposition::Transcript,
        };
        response.validate_against(request, expected_model)?;
        Ok(response)
    }

    /// Marks a verified local adapter result as an intentional no-speech
    /// completion. Generic empty responses remain normal responses and are
    /// treated as a recoverable inference failure by the state layer.
    pub fn no_speech() -> Self {
        Self {
            emissions: Vec::new(),
            disposition: AsrResponseDisposition::NoSpeech,
        }
    }

    pub fn is_no_speech(&self) -> bool {
        self.disposition == AsrResponseDisposition::NoSpeech
    }

    /// Validates an adapter response even when the adapter constructed this
    /// public transport type directly instead of using [`Self::new`].
    pub fn validate_against(
        &self,
        request: &AsrRequest,
        expected_model: &ModelProvenance,
    ) -> Result<(), String> {
        request.validate()?;
        expected_model.validate()?;
        if self.emissions.len() > MAX_ASR_EMISSIONS_PER_REQUEST {
            return Err(format!(
                "ASR response exceeds {MAX_ASR_EMISSIONS_PER_REQUEST} emissions"
            ));
        }
        if self.is_no_speech() && !self.emissions.is_empty() {
            return Err("a no-speech ASR response cannot include transcript emissions".to_owned());
        }

        let mut revisions = BTreeMap::<String, (u32, bool)>::new();
        for emission in &self.emissions {
            emission.validate()?;
            if &emission.model_provenance != expected_model {
                return Err("ASR emission model provenance does not match its engine".to_owned());
            }
            if emission.capture_start_ns < request.audio.capture_start_ns()
                || emission.capture_end_ns > request.audio.capture_end_ns()
            {
                return Err("ASR emission must remain inside its requested audio window".to_owned());
            }
            if !request.emit_partials && emission.kind == TranscriptEmissionKind::Partial {
                return Err("ASR request does not permit partial emissions".to_owned());
            }

            let history = revisions
                .entry(emission.utterance_key.clone())
                .or_insert((0, false));
            if history.1 {
                return Err("an utterance cannot emit after its final revision".to_owned());
            }
            if emission.revision <= history.0 {
                return Err("transcript emission revisions must increase".to_owned());
            }
            history.0 = emission.revision;
            history.1 = emission.kind == TranscriptEmissionKind::Final;
        }

        Ok(())
    }
}

/// Stable identity for a final native ASR emission. The SQLite store binds
/// this to the durable transcript revision in the same transaction, so a
/// process crash cannot turn a replay into a second transcript record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AsrFinalIdempotencyKey {
    pub session_id: Uuid,
    pub utterance_key: String,
    pub emission_revision: u32,
}

impl AsrFinalIdempotencyKey {
    pub fn new(
        session_id: Uuid,
        utterance_key: impl Into<String>,
        emission_revision: u32,
    ) -> Result<Self, String> {
        let key = Self {
            session_id,
            utterance_key: utterance_key.into(),
            emission_revision,
        };
        validate_utterance_key(&key.utterance_key)?;
        if key.emission_revision == 0 {
            return Err("final ASR emission revision must start at 1".to_owned());
        }
        Ok(key)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_utterance_key(&self.utterance_key)?;
        if self.emission_revision == 0 {
            return Err("final ASR emission revision must start at 1".to_owned());
        }
        Ok(())
    }

    /// Returns the opaque durable representation of the adapter's utterance
    /// key. Adapter keys are not guaranteed to be non-sensitive, so SQLite
    /// never stores their raw value.
    pub(crate) fn opaque_utterance_key_sha256(&self) -> String {
        opaque_utterance_key_sha256(&self.utterance_key)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransientTranscriptEmission {
    pub session_id: Uuid,
    pub logical_span_id: Uuid,
    pub emission: TranscriptEmission,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalTranscriptEmission {
    pub session_id: Uuid,
    pub logical_span_id: Uuid,
    pub emission: TranscriptEmission,
    reservation_id: Uuid,
}

impl FinalTranscriptEmission {
    pub(crate) fn idempotency_key(&self) -> AsrFinalIdempotencyKey {
        AsrFinalIdempotencyKey::new(
            self.session_id,
            self.emission.utterance_key.clone(),
            self.emission.revision,
        )
        .expect("mapped final emissions are already validated")
    }

    /// A deterministic fingerprint of all final-ASR fields, including
    /// word-level timing that is intentionally not copied into the compact
    /// transcript revision. It lets SQLite distinguish a harmless replay from
    /// a conflicting reuse of the same final-emission key.
    pub(crate) fn idempotency_payload_sha256(&self) -> String {
        let bytes = serde_json::to_vec(&self.emission)
            .expect("a validated transcript emission serializes for idempotency");
        let digest = Sha256::digest(bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// A local-only result of mapping an ASR emission into application state.
///
/// Partial values intentionally remain transient. A final value carries the
/// same logical span ID allocated for any preceding partial, so the native
/// persistence layer can create the first durable revision without retaining
/// the partial text.
#[derive(Clone, Debug, PartialEq)]
pub enum MappedTranscriptEmission {
    Partial(TransientTranscriptEmission),
    Final(FinalTranscriptEmission),
    /// A duplicate final is already pending, so it must not be persisted again.
    /// Durable replay detection belongs to the SQLite idempotency key. Duplicate
    /// partials are also ignored when they exactly match the latest accepted
    /// transient emission.
    Ignored,
}

#[derive(Clone, Debug)]
struct UtteranceState {
    logical_span_id: Uuid,
    last_accepted_emission: Option<TranscriptEmission>,
    pending_final: Option<FinalTranscriptEmission>,
}

/// Tracks only in-flight ASR utterances inside Rust. Completed finals are
/// released after SQLite commits; durable replay detection lives in the local
/// store. This prevents long recordings from retaining all transcript text in
/// process memory.
#[derive(Debug, Default)]
pub struct TranscriptEmissionMapper {
    utterances: BTreeMap<(Uuid, String), UtteranceState>,
}

impl TranscriptEmissionMapper {
    pub fn map(
        &mut self,
        session_id: Uuid,
        emission: TranscriptEmission,
    ) -> Result<MappedTranscriptEmission, String> {
        emission.validate()?;
        let key = (session_id, emission.utterance_key.clone());
        if !self.utterances.contains_key(&key) {
            if self.utterances.len() >= MAX_TRANSIENT_ASR_UTTERANCES {
                return Err(format!(
                    "local ASR has reached its {MAX_TRANSIENT_ASR_UTTERANCES}-utterance transient limit"
                ));
            }
            self.utterances.insert(
                key.clone(),
                UtteranceState {
                    logical_span_id: logical_span_id_for_asr_utterance(
                        session_id,
                        &emission.utterance_key,
                    ),
                    last_accepted_emission: None,
                    pending_final: None,
                },
            );
        }
        let state = self
            .utterances
            .get_mut(&key)
            .expect("a just-inserted ASR utterance state exists");

        if let Some(pending_final) = &state.pending_final {
            if pending_final.emission == emission {
                return Ok(MappedTranscriptEmission::Ignored);
            }
            return Err("an utterance final persistence is already pending".to_owned());
        }

        if let Some(accepted) = &state.last_accepted_emission {
            if emission.revision <= accepted.revision {
                if accepted == &emission {
                    return Ok(MappedTranscriptEmission::Ignored);
                }
                return Err("transcript emission revisions must increase".to_owned());
            }
        }

        let logical_span_id = state.logical_span_id;
        match emission.kind {
            TranscriptEmissionKind::Partial => {
                state.last_accepted_emission = Some(emission.clone());
                Ok(MappedTranscriptEmission::Partial(
                    TransientTranscriptEmission {
                        session_id,
                        logical_span_id,
                        emission,
                    },
                ))
            }
            TranscriptEmissionKind::Final => {
                let final_emission = FinalTranscriptEmission {
                    session_id,
                    logical_span_id,
                    emission,
                    reservation_id: Uuid::new_v4(),
                };
                state.pending_final = Some(final_emission.clone());
                Ok(MappedTranscriptEmission::Final(final_emission))
            }
        }
    }

    /// Releases a completed final after its revision and audit event commit.
    /// SQLite owns durable replay suppression, so the mapper retains no final
    /// text after this point.
    pub(crate) fn commit_final(&mut self, final_emission: &FinalTranscriptEmission) {
        let key = (
            final_emission.session_id,
            final_emission.emission.utterance_key.clone(),
        );
        let state = self
            .utterances
            .get(&key)
            .expect("a mapped final transcript reservation must remain registered");
        assert!(
            state.pending_final.as_ref() == Some(final_emission),
            "a final transcript reservation must remain pending until it commits"
        );

        self.utterances.remove(&key);
    }

    /// Releases a final reservation after persistence fails, leaving any
    /// earlier partial revision intact so the same final can be retried.
    pub(crate) fn abort_final(
        &mut self,
        final_emission: &FinalTranscriptEmission,
    ) -> Result<(), String> {
        let key = (
            final_emission.session_id,
            final_emission.emission.utterance_key.clone(),
        );
        let state = self
            .utterances
            .get_mut(&key)
            .ok_or_else(|| "final transcript reservation is unknown".to_owned())?;
        if state.pending_final.as_ref() != Some(final_emission) {
            return Err("final transcript reservation is no longer pending".to_owned());
        }

        state.pending_final = None;
        Ok(())
    }

    /// End a session only after all final reservations have either committed
    /// or aborted. This drops any remaining transient partial text.
    pub(crate) fn clear_session(&mut self, session_id: Uuid) -> Result<(), String> {
        if self
            .utterances
            .iter()
            .any(|((stored_session_id, _), state)| {
                *stored_session_id == session_id && state.pending_final.is_some()
            })
        {
            return Err("cannot clear an ASR session with a pending final reservation".to_owned());
        }
        self.utterances
            .retain(|(stored_session_id, _), _| *stored_session_id != session_id);
        Ok(())
    }

    #[cfg(test)]
    fn session_utterance_count(&self, session_id: Uuid) -> usize {
        self.utterances
            .keys()
            .filter(|(stored_session_id, _)| *stored_session_id == session_id)
            .count()
    }
}

/// A local ASR adapter. Implementors emit only bounded, provenance-carrying
/// records and never own egress authority.
pub trait AsrEngine: InferenceEngine {
    fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError>;
}

fn validate_utterance_key(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("transcript utterance key must not be empty".to_owned());
    }
    if value.len() > MAX_MODEL_IDENTIFIER_BYTES {
        return Err(format!(
            "transcript utterance key exceeds {MAX_MODEL_IDENTIFIER_BYTES} bytes"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err("transcript utterance key must not contain control characters".to_owned());
    }
    Ok(())
}

/// Derives the only logical transcript ID accepted for a native ASR
/// utterance. The audit store rechecks this after reopening SQLite so an
/// idempotency key cannot be rebound to a different transcript span.
pub(crate) fn logical_span_id_for_asr_utterance(session_id: Uuid, utterance_key: &str) -> Uuid {
    logical_span_id_for_asr_utterance_digest(
        session_id,
        &opaque_utterance_key_sha256(utterance_key),
    )
}

/// Recomputes the stable logical transcript ID from the opaque utterance-key
/// representation retained by SQLite and audit verification.
pub(crate) fn logical_span_id_for_asr_utterance_digest(
    session_id: Uuid,
    utterance_key_sha256: &str,
) -> Uuid {
    let mut digest = Sha256::new();
    digest.update(b"word-covenant/transcript-logical-span/v1\0");
    digest.update(session_id.as_bytes());
    digest.update(utterance_key_sha256.as_bytes());
    let digest = digest.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // Mark the derived value as an RFC 4122 name-based UUID without relying
    // on uuid's optional v5 feature.
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn opaque_utterance_key_sha256(utterance_key: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"word-covenant/asr-utterance-key/v1\0");
    digest.update(utterance_key.as_bytes());
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceAudioWindow, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ};
    use uuid::Uuid;

    fn model() -> ModelProvenance {
        ModelProvenance::new("fixture", "fixture-asr", "v1", "a".repeat(64)).unwrap()
    }

    fn request() -> AsrRequest {
        AsrRequest::new(
            InferenceAudioWindow::new(
                Uuid::nil(),
                0,
                1_000_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.0; 16_000],
            )
            .unwrap(),
            Some("zh".to_owned()),
            true,
        )
        .unwrap()
    }

    fn emission(end_ns: u64) -> TranscriptEmission {
        TranscriptEmission {
            utterance_key: "fixture-utterance-1".to_owned(),
            capture_start_ns: 0,
            capture_end_ns: end_ns,
            text: "本机记录".to_owned(),
            kind: TranscriptEmissionKind::Final,
            revision: 1,
            word_timings: Vec::new(),
            model_provenance: model(),
        }
    }

    #[test]
    fn rejects_emissions_outside_the_requested_window() {
        let error =
            AsrResponse::new(&request(), &model(), vec![emission(1_000_000_001)]).unwrap_err();

        assert!(error.contains("requested audio window"));
    }

    #[test]
    fn serializes_partial_and_final_kinds_distinctly() {
        assert_eq!(
            serde_json::to_value(TranscriptEmissionKind::Partial).unwrap(),
            serde_json::Value::String("partial".to_owned())
        );
        assert_eq!(
            serde_json::to_value(TranscriptEmissionKind::Final).unwrap(),
            serde_json::Value::String("final".to_owned())
        );
    }

    #[test]
    fn rejects_a_revision_after_finalization() {
        let mut partial = emission(500_000_000);
        partial.kind = TranscriptEmissionKind::Partial;
        partial.revision = 1;
        let mut final_emission = emission(1_000_000_000);
        final_emission.revision = 2;
        let mut after_final = emission(1_000_000_000);
        after_final.revision = 3;

        let error = AsrResponse::new(
            &request(),
            &model(),
            vec![partial, final_emission, after_final],
        )
        .unwrap_err();

        assert!(error.contains("after its final"));
    }

    #[test]
    fn maps_partial_and_final_emissions_to_one_durable_logical_span() {
        let session_id = Uuid::new_v4();
        let mut mapper = TranscriptEmissionMapper::default();
        let mut partial = emission(500_000_000);
        partial.kind = TranscriptEmissionKind::Partial;
        partial.revision = 1;
        let mut final_emission = emission(1_000_000_000);
        final_emission.revision = 2;

        let partial = match mapper.map(session_id, partial).unwrap() {
            MappedTranscriptEmission::Partial(partial) => partial,
            MappedTranscriptEmission::Final(_) => panic!("partial emission became durable"),
            MappedTranscriptEmission::Ignored => panic!("first partial emission was ignored"),
        };
        let final_emission = match mapper.map(session_id, final_emission).unwrap() {
            MappedTranscriptEmission::Final(final_emission) => final_emission,
            MappedTranscriptEmission::Partial(_) => panic!("final emission stayed transient"),
            MappedTranscriptEmission::Ignored => panic!("first final emission was ignored"),
        };

        assert_eq!(partial.logical_span_id, final_emission.logical_span_id);
        assert_eq!(final_emission.emission.revision, 2);
        mapper.commit_final(&final_emission);
        assert_eq!(mapper.session_utterance_count(session_id), 0);

        // After a successful SQLite commit the mapper deliberately forgets
        // final text. A replay is mapped with the same durable identity; the
        // SQLite idempotency table, rather than process memory, suppresses it.
        let replay = match mapper
            .map(session_id, final_emission.emission.clone())
            .unwrap()
        {
            MappedTranscriptEmission::Final(replay) => replay,
            MappedTranscriptEmission::Partial(_) => panic!("replayed final became transient"),
            MappedTranscriptEmission::Ignored => panic!("completed final remained in mapper"),
        };
        assert_eq!(replay.logical_span_id, final_emission.logical_span_id);
        assert_eq!(replay.idempotency_key(), final_emission.idempotency_key());
        mapper.commit_final(&replay);
        assert_eq!(mapper.session_utterance_count(session_id), 0);
    }

    #[test]
    fn releases_a_reserved_final_for_retry_and_forgets_it_after_commit() {
        let session_id = Uuid::new_v4();
        let mut mapper = TranscriptEmissionMapper::default();
        let final_emission = emission(1_000_000_000);

        let reservation = match mapper.map(session_id, final_emission.clone()).unwrap() {
            MappedTranscriptEmission::Final(final_emission) => final_emission,
            MappedTranscriptEmission::Partial(_) => panic!("final emission stayed transient"),
            MappedTranscriptEmission::Ignored => panic!("first final emission was ignored"),
        };
        assert!(matches!(
            mapper.map(session_id, final_emission.clone()).unwrap(),
            MappedTranscriptEmission::Ignored
        ));

        mapper.abort_final(&reservation).unwrap();
        let retry = match mapper.map(session_id, final_emission.clone()).unwrap() {
            MappedTranscriptEmission::Final(final_emission) => final_emission,
            MappedTranscriptEmission::Partial(_) => panic!("retry final emission stayed transient"),
            MappedTranscriptEmission::Ignored => panic!("retry final emission was ignored"),
        };
        mapper.commit_final(&retry);

        let replay = match mapper.map(session_id, final_emission).unwrap() {
            MappedTranscriptEmission::Final(replay) => replay,
            MappedTranscriptEmission::Partial(_) => panic!("replayed final became transient"),
            MappedTranscriptEmission::Ignored => panic!("completed final remained in mapper"),
        };
        assert_eq!(replay.idempotency_key(), retry.idempotency_key());
        mapper.commit_final(&replay);
        assert_eq!(mapper.session_utterance_count(session_id), 0);
    }

    #[test]
    fn maps_a_final_without_emitting_its_optional_partial() {
        let session_id = Uuid::new_v4();
        let mut mapper = TranscriptEmissionMapper::default();
        let mut final_emission = emission(1_000_000_000);
        final_emission.revision = 2;

        let mapped = mapper.map(session_id, final_emission).unwrap();
        assert!(matches!(mapped, MappedTranscriptEmission::Final(_)));
    }
}
