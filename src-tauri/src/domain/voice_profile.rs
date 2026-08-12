use crate::diarization::{SpeakerEmbedding, SpeakerSampleQuality};
use crate::domain::speaker::validate_speaker_cluster_id;
use crate::inference::ModelProvenance;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const MAX_VOICE_PROFILE_NAME_LENGTH: usize = 80;
pub const READY_CONFIRMED_DURATION_NS: u64 = 4_000_000_000;
pub const MAX_CONFIRMED_DURATION_NS: u64 = 30_000_000_000;
pub const MAX_PROTOTYPES_PER_PROFILE: usize = 16;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceProfileState {
    Learning,
    Ready,
    RelearnRequired,
}

/// One immutable revision of a persistent local voice profile.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceProfile {
    pub id: Uuid,
    pub revision_id: Uuid,
    pub parent_revision_id: Option<Uuid>,
    pub revision: u32,
    pub display_name: String,
    pub state: VoiceProfileState,
    pub model: ModelProvenance,
    pub confirmed_duration_ns: u64,
    pub learning_started_at: DateTime<Utc>,
    pub origin_session_id: Option<Uuid>,
    pub origin_cluster_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl VoiceProfile {
    pub fn new(display_name: impl Into<String>, model: ModelProvenance) -> Result<Self, String> {
        Self::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            display_name,
            model,
            Utc::now(),
        )
    }

    pub fn new_with_id(
        id: Uuid,
        revision_id: Uuid,
        display_name: impl Into<String>,
        model: ModelProvenance,
        created_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        Self::new_with_origin(id, revision_id, display_name, model, None, None, created_at)
    }

    pub fn new_from_cluster_with_id(
        id: Uuid,
        revision_id: Uuid,
        display_name: impl Into<String>,
        model: ModelProvenance,
        origin_session_id: Uuid,
        origin_cluster_id: impl Into<String>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        Self::new_with_origin(
            id,
            revision_id,
            display_name,
            model,
            Some(origin_session_id),
            Some(origin_cluster_id.into()),
            created_at,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_origin(
        id: Uuid,
        revision_id: Uuid,
        display_name: impl Into<String>,
        model: ModelProvenance,
        origin_session_id: Option<Uuid>,
        origin_cluster_id: Option<String>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let profile = Self {
            id,
            revision_id,
            parent_revision_id: None,
            revision: 1,
            display_name: normalize_profile_name(display_name.into())?,
            state: VoiceProfileState::Learning,
            model,
            confirmed_duration_ns: 0,
            learning_started_at: created_at,
            origin_session_id,
            origin_cluster_id,
            created_at,
            updated_at: created_at,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn rename_with_id(
        previous: &Self,
        revision_id: Uuid,
        display_name: impl Into<String>,
        updated_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let mut revision = previous.successor(revision_id, updated_at)?;
        revision.display_name = normalize_profile_name(display_name.into())?;
        revision.validate_successor_of(previous)?;
        Ok(revision)
    }

    pub fn confirm_with_id(
        previous: &Self,
        revision_id: Uuid,
        confirmed_duration_ns: u64,
        updated_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        if confirmed_duration_ns == 0 || confirmed_duration_ns > MAX_CONFIRMED_DURATION_NS {
            return Err("confirmed speaker duration is outside the supported bounds".to_owned());
        }
        let mut revision = previous.successor(revision_id, updated_at)?;
        revision.confirmed_duration_ns = previous
            .confirmed_duration_ns
            .saturating_add(confirmed_duration_ns)
            .min(MAX_CONFIRMED_DURATION_NS);
        revision.state = if revision.confirmed_duration_ns >= READY_CONFIRMED_DURATION_NS {
            VoiceProfileState::Ready
        } else {
            VoiceProfileState::Learning
        };
        revision.validate_successor_of(previous)?;
        Ok(revision)
    }

    pub fn require_relearn_with_id(
        previous: &Self,
        revision_id: Uuid,
        model: ModelProvenance,
        updated_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let mut revision = previous.successor(revision_id, updated_at)?;
        revision.model = model;
        revision.confirmed_duration_ns = 0;
        revision.state = VoiceProfileState::RelearnRequired;
        revision.learning_started_at = updated_at;
        revision.validate_successor_of(previous)?;
        Ok(revision)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_uuid(self.id, "voice profile ID")?;
        validate_uuid(self.revision_id, "voice profile revision ID")?;
        validate_revision_shape(self.parent_revision_id, self.revision)?;
        validate_profile_name(&self.display_name)?;
        self.model.validate()?;
        if self.updated_at < self.created_at {
            return Err("voice profile update time precedes its creation".to_owned());
        }
        if self.learning_started_at < self.created_at || self.learning_started_at > self.updated_at
        {
            return Err("voice profile learning start time is outside its lifetime".to_owned());
        }
        if self.confirmed_duration_ns > MAX_CONFIRMED_DURATION_NS {
            return Err("voice profile confirmed duration exceeds its bound".to_owned());
        }
        match (&self.origin_session_id, &self.origin_cluster_id) {
            (Some(session_id), Some(cluster_id)) => {
                validate_uuid(*session_id, "voice profile origin session ID")?;
                validate_speaker_cluster_id(cluster_id)?;
            }
            (None, None) => {}
            _ => {
                return Err(
                    "voice profile origin session and cluster must be stored together".to_owned(),
                )
            }
        }
        match self.state {
            VoiceProfileState::Learning
                if self.confirmed_duration_ns >= READY_CONFIRMED_DURATION_NS =>
            {
                Err("a learning voice profile already has enough confirmed audio".to_owned())
            }
            VoiceProfileState::Ready
                if self.confirmed_duration_ns < READY_CONFIRMED_DURATION_NS =>
            {
                Err("a ready voice profile lacks confirmed audio".to_owned())
            }
            VoiceProfileState::RelearnRequired if self.confirmed_duration_ns != 0 => {
                Err("a voice profile requiring relearning must reset confirmed duration".to_owned())
            }
            _ => Ok(()),
        }
    }

    pub fn validate_successor_of(&self, previous: &Self) -> Result<(), String> {
        previous.validate()?;
        self.validate()?;
        if self.id != previous.id
            || self.parent_revision_id != Some(previous.revision_id)
            || self.revision
                != previous
                    .revision
                    .checked_add(1)
                    .ok_or("revision overflow")?
            || self.created_at != previous.created_at
            || self.origin_session_id != previous.origin_session_id
            || self.origin_cluster_id != previous.origin_cluster_id
        {
            return Err("voice profile revision does not immediately follow its parent".to_owned());
        }
        if self.updated_at < previous.updated_at {
            return Err("voice profile revision time moved backwards".to_owned());
        }
        Ok(())
    }

    fn successor(&self, revision_id: Uuid, updated_at: DateTime<Utc>) -> Result<Self, String> {
        self.validate()?;
        validate_uuid(revision_id, "voice profile revision ID")?;
        if revision_id == self.revision_id {
            return Err("voice profile revisions must use distinct IDs".to_owned());
        }
        Ok(Self {
            id: self.id,
            revision_id,
            parent_revision_id: Some(self.revision_id),
            revision: self.revision.checked_add(1).ok_or("revision overflow")?,
            display_name: self.display_name.clone(),
            state: self.state,
            model: self.model.clone(),
            confirmed_duration_ns: self.confirmed_duration_ns,
            learning_started_at: self.learning_started_at,
            origin_session_id: self.origin_session_id,
            origin_cluster_id: self.origin_cluster_id.clone(),
            created_at: self.created_at,
            updated_at,
        })
    }
}

/// One bounded, user-confirmed prototype. Automatic matches never construct
/// this value on their own.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerPrototype {
    pub id: Uuid,
    pub profile_id: Uuid,
    pub profile_revision_id: Uuid,
    pub embedding: SpeakerEmbedding,
    pub confirmed_duration_ns: u64,
    pub confirmed_at: DateTime<Utc>,
    pub source_observation_id: Option<Uuid>,
}

impl SpeakerPrototype {
    pub fn new_with_id(
        id: Uuid,
        profile: &VoiceProfile,
        embedding: SpeakerEmbedding,
        confirmed_duration_ns: u64,
        confirmed_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        Self::new_with_source(
            id,
            profile,
            embedding,
            confirmed_duration_ns,
            confirmed_at,
            None,
        )
    }

    pub fn new_from_observation_with_id(
        id: Uuid,
        profile: &VoiceProfile,
        embedding: SpeakerEmbedding,
        confirmed_duration_ns: u64,
        confirmed_at: DateTime<Utc>,
        source_observation_id: Uuid,
    ) -> Result<Self, String> {
        Self::new_with_source(
            id,
            profile,
            embedding,
            confirmed_duration_ns,
            confirmed_at,
            Some(source_observation_id),
        )
    }

    fn new_with_source(
        id: Uuid,
        profile: &VoiceProfile,
        embedding: SpeakerEmbedding,
        confirmed_duration_ns: u64,
        confirmed_at: DateTime<Utc>,
        source_observation_id: Option<Uuid>,
    ) -> Result<Self, String> {
        let prototype = Self {
            id,
            profile_id: profile.id,
            profile_revision_id: profile.revision_id,
            embedding,
            confirmed_duration_ns,
            confirmed_at,
            source_observation_id,
        };
        prototype.validate_for_profile(profile)?;
        Ok(prototype)
    }

    pub fn validate_for_profile(&self, profile: &VoiceProfile) -> Result<(), String> {
        validate_uuid(self.id, "speaker prototype ID")?;
        profile.validate()?;
        if self.profile_id != profile.id || self.profile_revision_id != profile.revision_id {
            return Err("speaker prototype is not bound to its profile revision".to_owned());
        }
        if self.embedding.model() != &profile.model {
            return Err("speaker prototype uses an incompatible model space".to_owned());
        }
        if self.confirmed_duration_ns == 0 || self.confirmed_duration_ns > MAX_CONFIRMED_DURATION_NS
        {
            return Err("speaker prototype confirmed duration is outside bounds".to_owned());
        }
        if self.confirmed_at < profile.created_at || self.confirmed_at > profile.updated_at {
            return Err(
                "speaker prototype confirmation time is outside its profile revision".to_owned(),
            );
        }
        if self.source_observation_id == Some(Uuid::nil()) {
            return Err("speaker prototype source observation ID must not be nil".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerObservationDecision {
    MatchedProfile,
    AnonymousCluster,
    Unknown,
    Ambiguous,
    Ineligible,
}

/// Immutable local evidence for one final transcript span. It contains no PCM.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerObservation {
    pub id: Uuid,
    pub session_id: Uuid,
    pub transcript_revision_id: Uuid,
    pub profile_id: Option<Uuid>,
    pub anonymous_cluster_id: Option<String>,
    pub label_snapshot: Option<String>,
    pub decision: SpeakerObservationDecision,
    pub similarity: Option<f32>,
    pub runner_up_similarity: Option<f32>,
    pub embedding: SpeakerEmbedding,
    pub quality: SpeakerSampleQuality,
    pub observed_at: DateTime<Utc>,
}

impl SpeakerObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: Uuid,
        session_id: Uuid,
        transcript_revision_id: Uuid,
        profile_id: Option<Uuid>,
        anonymous_cluster_id: Option<String>,
        label_snapshot: Option<String>,
        decision: SpeakerObservationDecision,
        similarity: Option<f32>,
        runner_up_similarity: Option<f32>,
        embedding: SpeakerEmbedding,
        quality: SpeakerSampleQuality,
        observed_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let observation = Self {
            id,
            session_id,
            transcript_revision_id,
            profile_id,
            anonymous_cluster_id,
            label_snapshot,
            decision,
            similarity,
            runner_up_similarity,
            embedding,
            quality,
            observed_at,
        };
        observation.validate()?;
        Ok(observation)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_uuid(self.id, "speaker observation ID")?;
        validate_uuid(self.session_id, "speaker observation session ID")?;
        validate_uuid(
            self.transcript_revision_id,
            "speaker observation transcript revision ID",
        )?;
        if self.profile_id == Some(Uuid::nil()) {
            return Err("speaker observation profile ID must not be nil".to_owned());
        }
        if let Some(cluster_id) = &self.anonymous_cluster_id {
            if !cluster_id.starts_with("speaker-") || cluster_id.len() > 96 {
                return Err("speaker observation has an invalid anonymous cluster ID".to_owned());
            }
        }
        if let Some(label) = &self.label_snapshot {
            validate_profile_name(label)?;
        }
        for (label, value) in [
            ("similarity", self.similarity),
            ("runner-up similarity", self.runner_up_similarity),
        ] {
            if value.is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value)) {
                return Err(format!("speaker observation {label} is outside bounds"));
            }
        }
        match self.decision {
            SpeakerObservationDecision::MatchedProfile
                if self.profile_id.is_none()
                    || self.label_snapshot.is_none()
                    || self.anonymous_cluster_id.is_some()
                    || self.similarity.is_none() =>
            {
                return Err("matched speaker observation lacks profile evidence".to_owned());
            }
            SpeakerObservationDecision::AnonymousCluster
                if self.anonymous_cluster_id.is_none() || self.profile_id.is_some() =>
            {
                return Err("anonymous speaker observation has invalid assignments".to_owned());
            }
            SpeakerObservationDecision::Unknown
            | SpeakerObservationDecision::Ambiguous
            | SpeakerObservationDecision::Ineligible
                if self.profile_id.is_some() || self.anonymous_cluster_id.is_some() =>
            {
                return Err("unassigned speaker observation contains an assignment".to_owned());
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileAuditBinding {
    pub profile_id: Uuid,
    pub revision_id: Uuid,
    pub parent_revision_id: Option<Uuid>,
    pub revision: u32,
    pub display_name_sha256: String,
    pub state: VoiceProfileState,
    pub model: ModelProvenance,
    pub confirmed_duration_ns: u64,
    pub learning_started_at: DateTime<Utc>,
    pub origin_session_id: Option<Uuid>,
    pub origin_cluster_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl VoiceProfileAuditBinding {
    pub fn from_profile(profile: &VoiceProfile) -> Result<Self, String> {
        profile.validate()?;
        Ok(Self {
            profile_id: profile.id,
            revision_id: profile.revision_id,
            parent_revision_id: profile.parent_revision_id,
            revision: profile.revision,
            display_name_sha256: sha256_hex(profile.display_name.as_bytes()),
            state: profile.state,
            model: profile.model.clone(),
            confirmed_duration_ns: profile.confirmed_duration_ns,
            learning_started_at: profile.learning_started_at,
            origin_session_id: profile.origin_session_id,
            origin_cluster_id: profile.origin_cluster_id.clone(),
            created_at: profile.created_at,
            updated_at: profile.updated_at,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerPrototypeAuditBinding {
    pub prototype_id: Uuid,
    pub profile_id: Uuid,
    pub profile_revision_id: Uuid,
    pub model: ModelProvenance,
    pub dimensions: usize,
    pub embedding_sha256: String,
    pub confirmed_duration_ns: u64,
    pub confirmed_at: DateTime<Utc>,
    pub source_observation_id: Option<Uuid>,
}

impl SpeakerPrototypeAuditBinding {
    pub fn from_prototype(prototype: &SpeakerPrototype) -> Self {
        Self {
            prototype_id: prototype.id,
            profile_id: prototype.profile_id,
            profile_revision_id: prototype.profile_revision_id,
            model: prototype.embedding.model().clone(),
            dimensions: prototype.embedding.dimensions(),
            embedding_sha256: embedding_sha256(&prototype.embedding),
            confirmed_duration_ns: prototype.confirmed_duration_ns,
            confirmed_at: prototype.confirmed_at,
            source_observation_id: prototype.source_observation_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerObservationAuditBinding {
    pub observation_id: Uuid,
    pub session_id: Uuid,
    pub transcript_revision_id: Uuid,
    pub profile_id: Option<Uuid>,
    pub anonymous_cluster_id: Option<String>,
    pub label_snapshot_sha256: Option<String>,
    pub decision: SpeakerObservationDecision,
    pub similarity_bits: Option<u32>,
    pub runner_up_similarity_bits: Option<u32>,
    pub model: ModelProvenance,
    pub dimensions: usize,
    pub embedding_sha256: String,
    pub voiced_duration_ns: u64,
    pub voiced_ratio_bits: u32,
    pub signal_quality_bits: u32,
    pub overlap_probability_bits: u32,
    pub observed_at: DateTime<Utc>,
}

impl SpeakerObservationAuditBinding {
    pub fn from_observation(observation: &SpeakerObservation) -> Result<Self, String> {
        observation.validate()?;
        Ok(Self {
            observation_id: observation.id,
            session_id: observation.session_id,
            transcript_revision_id: observation.transcript_revision_id,
            profile_id: observation.profile_id,
            anonymous_cluster_id: observation.anonymous_cluster_id.clone(),
            label_snapshot_sha256: observation
                .label_snapshot
                .as_ref()
                .map(|label| sha256_hex(label.as_bytes())),
            decision: observation.decision,
            similarity_bits: observation.similarity.map(f32::to_bits),
            runner_up_similarity_bits: observation.runner_up_similarity.map(f32::to_bits),
            model: observation.embedding.model().clone(),
            dimensions: observation.embedding.dimensions(),
            embedding_sha256: embedding_sha256(&observation.embedding),
            voiced_duration_ns: observation.quality.voiced_duration_ns(),
            voiced_ratio_bits: observation.quality.voiced_ratio().to_bits(),
            signal_quality_bits: observation.quality.signal_quality().to_bits(),
            overlap_probability_bits: observation.quality.overlap_probability().to_bits(),
            observed_at: observation.observed_at,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileCreatedAuditPayload {
    pub profile: VoiceProfileAuditBinding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileRevisionAuditPayload {
    pub profile: VoiceProfileAuditBinding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileEnrollmentAuditPayload {
    pub profile: VoiceProfileAuditBinding,
    pub prototype: SpeakerPrototypeAuditBinding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileDeletedAuditPayload {
    pub profile_id: Uuid,
    pub purged_audit_event_ids_sha256: String,
    pub purged_audit_event_count: usize,
}

impl VoiceProfileDeletedAuditPayload {
    pub fn new(profile_id: Uuid, purged_audit_event_ids: &[Uuid]) -> Result<Self, String> {
        validate_uuid(profile_id, "voice profile deletion ID")?;
        if purged_audit_event_ids.is_empty() || purged_audit_event_ids.contains(&Uuid::nil()) {
            return Err("voice profile deletion must bind non-empty audit event IDs".to_owned());
        }
        let mut event_ids = purged_audit_event_ids.to_vec();
        event_ids.sort_unstable();
        event_ids.dedup();
        if event_ids.len() != purged_audit_event_ids.len() {
            return Err("voice profile deletion contains duplicate audit event IDs".to_owned());
        }
        let material = event_ids
            .iter()
            .map(Uuid::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Self {
            profile_id,
            purged_audit_event_ids_sha256: sha256_hex(material.as_bytes()),
            purged_audit_event_count: event_ids.len(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerObservationAuditPayload {
    pub observation: SpeakerObservationAuditBinding,
}

pub fn embedding_bytes(embedding: &SpeakerEmbedding) -> Vec<u8> {
    embedding
        .values()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn embedding_sha256(embedding: &SpeakerEmbedding) -> String {
    sha256_hex(&embedding_bytes(embedding))
}

fn sha256_hex(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn normalize_profile_name(value: String) -> Result<String, String> {
    let normalized = value.trim().to_owned();
    validate_profile_name(&normalized)?;
    Ok(normalized)
}

fn validate_profile_name(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("voice profile name must not be empty".to_owned());
    }
    if value.chars().count() > MAX_VOICE_PROFILE_NAME_LENGTH {
        return Err(format!(
            "voice profile name exceeds {MAX_VOICE_PROFILE_NAME_LENGTH} characters"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err("voice profile name must not contain control characters".to_owned());
    }
    Ok(())
}

fn validate_uuid(value: Uuid, label: &str) -> Result<(), String> {
    if value.is_nil() {
        Err(format!("{label} must not be nil"))
    } else {
        Ok(())
    }
}

fn validate_revision_shape(parent_revision_id: Option<Uuid>, revision: u32) -> Result<(), String> {
    if revision == 0 {
        return Err("voice profile revision must be greater than zero".to_owned());
    }
    if (revision == 1) != parent_revision_id.is_none() {
        return Err("voice profile revision has an invalid parent shape".to_owned());
    }
    if parent_revision_id == Some(Uuid::nil()) {
        return Err("voice profile parent revision ID must not be nil".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(version: &str) -> ModelProvenance {
        ModelProvenance::new("fixture", "speaker", version, "c".repeat(64)).unwrap()
    }

    fn embedding(version: &str) -> SpeakerEmbedding {
        SpeakerEmbedding::new(model(version), vec![1.0, 0.0, 0.0]).unwrap()
    }

    #[test]
    fn profile_moves_from_learning_to_ready_and_caps_confirmed_duration() {
        let created_at = Utc::now();
        let initial = VoiceProfile::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "  Alice  ",
            model("v1"),
            created_at,
        )
        .unwrap();
        assert_eq!(initial.display_name, "Alice");
        assert_eq!(initial.state, VoiceProfileState::Learning);

        let learning =
            VoiceProfile::confirm_with_id(&initial, Uuid::new_v4(), 2_000_000_000, created_at)
                .unwrap();
        let ready = VoiceProfile::confirm_with_id(
            &learning,
            Uuid::new_v4(),
            MAX_CONFIRMED_DURATION_NS,
            created_at,
        )
        .unwrap();
        assert_eq!(learning.state, VoiceProfileState::Learning);
        assert_eq!(ready.state, VoiceProfileState::Ready);
        assert_eq!(ready.confirmed_duration_ns, MAX_CONFIRMED_DURATION_NS);
    }

    #[test]
    fn model_change_requires_relearning_and_prototypes_cannot_cross_model_spaces() {
        let created_at = Utc::now();
        let initial = VoiceProfile::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "Alice",
            model("v1"),
            created_at,
        )
        .unwrap();
        let relearn = VoiceProfile::require_relearn_with_id(
            &initial,
            Uuid::new_v4(),
            model("v2"),
            created_at,
        )
        .unwrap();
        assert_eq!(relearn.state, VoiceProfileState::RelearnRequired);
        assert!(SpeakerPrototype::new_with_id(
            Uuid::new_v4(),
            &relearn,
            embedding("v1"),
            1_000_000_000,
            created_at,
        )
        .is_err());
    }

    #[test]
    fn audit_bindings_hash_names_and_vectors_instead_of_retaining_them() {
        let profile = VoiceProfile::new("Sensitive Name", model("v1")).unwrap();
        let serialized =
            serde_json::to_string(&VoiceProfileAuditBinding::from_profile(&profile).unwrap())
                .unwrap();
        assert!(!serialized.contains("Sensitive Name"));
        assert!(serialized.contains("displayNameSha256"));
    }
}
