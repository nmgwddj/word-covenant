use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const SPEAKER_CLUSTER_ID_PREFIX: &str = "speaker-";
pub const MAX_SPEAKER_LABEL_LENGTH: usize = 80;

/// Immutable, session-scoped anonymous speaker catalog entry.
///
/// The ID deliberately carries no identity information. The ordinal is only
/// used to generate the initial presentation label, such as `Speaker 1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerClusterRecord {
    pub id: String,
    pub session_id: Uuid,
    pub ordinal: u32,
}

impl SpeakerClusterRecord {
    pub fn new(session_id: Uuid, ordinal: u32) -> Result<Self, String> {
        Self::new_with_id(
            format!("{SPEAKER_CLUSTER_ID_PREFIX}{}", Uuid::new_v4()),
            session_id,
            ordinal,
        )
    }

    pub fn new_with_id(
        id: impl Into<String>,
        session_id: Uuid,
        ordinal: u32,
    ) -> Result<Self, String> {
        let record = Self {
            id: id.into(),
            session_id,
            ordinal,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn generated_label(&self) -> String {
        format!("Speaker {}", self.ordinal)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_speaker_cluster_id(&self.id)?;
        if self.session_id.is_nil() {
            return Err("speaker cluster session ID must not be nil".to_owned());
        }
        if self.ordinal == 0 {
            return Err("speaker cluster ordinal must be greater than zero".to_owned());
        }
        Ok(())
    }
}

/// One immutable display-label value for an anonymous speaker cluster.
///
/// Revision one is always the generated anonymous label. Later revisions are
/// user-entered presentation metadata, never an assertion of identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerClusterLabelRevision {
    pub id: Uuid,
    pub speaker_cluster_id: String,
    pub parent_revision_id: Option<Uuid>,
    pub revision: u32,
    pub label: String,
    pub is_user_named: bool,
}

impl SpeakerClusterLabelRevision {
    pub fn initial_generated(cluster: &SpeakerClusterRecord) -> Result<Self, String> {
        Self::initial_generated_with_id(Uuid::new_v4(), cluster)
    }

    pub fn initial_generated_with_id(
        id: Uuid,
        cluster: &SpeakerClusterRecord,
    ) -> Result<Self, String> {
        cluster.validate()?;
        Self::new_with_id(
            id,
            cluster.id.clone(),
            None,
            1,
            cluster.generated_label(),
            false,
        )
    }

    pub fn revision_of(previous: &Self, label: impl Into<String>) -> Result<Self, String> {
        Self::revision_of_with_id(Uuid::new_v4(), previous, label)
    }

    pub fn revision_of_with_id(
        id: Uuid,
        previous: &Self,
        label: impl Into<String>,
    ) -> Result<Self, String> {
        previous.validate()?;
        let revision = next_revision_number(previous.revision, "speaker label")?;
        let label = normalize_user_label(label.into())?;
        let next = Self::new_with_id(
            id,
            previous.speaker_cluster_id.clone(),
            Some(previous.id),
            revision,
            label,
            true,
        )?;
        next.validate_successor_of(previous)?;
        Ok(next)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_id(
        id: Uuid,
        speaker_cluster_id: impl Into<String>,
        parent_revision_id: Option<Uuid>,
        revision: u32,
        label: impl Into<String>,
        is_user_named: bool,
    ) -> Result<Self, String> {
        let record = Self {
            id,
            speaker_cluster_id: speaker_cluster_id.into(),
            parent_revision_id,
            revision,
            label: label.into(),
            is_user_named,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_revision_id(self.id, "speaker label revision")?;
        validate_speaker_cluster_id(&self.speaker_cluster_id)?;
        validate_revision_shape(
            self.parent_revision_id,
            self.revision,
            "speaker label revision",
        )?;
        validate_stored_label(&self.label)?;
        if !self.is_user_named && self.revision != 1 {
            return Err("only the initial speaker label revision may be generated".to_owned());
        }
        Ok(())
    }

    pub fn validate_for_cluster(&self, cluster: &SpeakerClusterRecord) -> Result<(), String> {
        cluster.validate()?;
        self.validate()?;
        if self.speaker_cluster_id != cluster.id {
            return Err("speaker label revision must belong to its speaker cluster".to_owned());
        }
        if !self.is_user_named && self.label != cluster.generated_label() {
            return Err("generated speaker label must match its anonymous ordinal".to_owned());
        }
        Ok(())
    }

    pub fn validate_successor_of(&self, previous: &Self) -> Result<(), String> {
        previous.validate()?;
        self.validate()?;
        if self.id == previous.id {
            return Err("speaker label revisions must use distinct IDs".to_owned());
        }
        if self.speaker_cluster_id != previous.speaker_cluster_id {
            return Err("speaker label revisions must retain their speaker cluster".to_owned());
        }
        if self.parent_revision_id != Some(previous.id) {
            return Err("speaker label revision must reference its immediate parent".to_owned());
        }
        if self.revision != next_revision_number(previous.revision, "speaker label")? {
            return Err("speaker label revision must increase by exactly one".to_owned());
        }
        Ok(())
    }
}

/// One immutable alias value for an anonymous speaker cluster.
///
/// An alias is a reversible merge. `None` is an explicit appended clearing
/// revision, not an in-place deletion of a prior merge.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerClusterAliasRevision {
    pub id: Uuid,
    pub speaker_cluster_id: String,
    pub parent_revision_id: Option<Uuid>,
    pub revision: u32,
    pub merged_into_cluster_id: Option<String>,
}

impl SpeakerClusterAliasRevision {
    pub fn aliased_to(
        speaker_cluster_id: impl Into<String>,
        merged_into_cluster_id: impl Into<String>,
    ) -> Result<Self, String> {
        Self::aliased_to_with_id(Uuid::new_v4(), speaker_cluster_id, merged_into_cluster_id)
    }

    pub fn aliased_to_with_id(
        id: Uuid,
        speaker_cluster_id: impl Into<String>,
        merged_into_cluster_id: impl Into<String>,
    ) -> Result<Self, String> {
        Self::new_with_id(
            id,
            speaker_cluster_id,
            None,
            1,
            Some(merged_into_cluster_id.into()),
        )
    }

    pub fn revision_of(
        previous: &Self,
        merged_into_cluster_id: Option<String>,
    ) -> Result<Self, String> {
        Self::revision_of_with_id(Uuid::new_v4(), previous, merged_into_cluster_id)
    }

    pub fn revision_of_with_id(
        id: Uuid,
        previous: &Self,
        merged_into_cluster_id: Option<String>,
    ) -> Result<Self, String> {
        previous.validate()?;
        let revision = next_revision_number(previous.revision, "speaker alias")?;
        let next = Self::new_with_id(
            id,
            previous.speaker_cluster_id.clone(),
            Some(previous.id),
            revision,
            merged_into_cluster_id,
        )?;
        next.validate_successor_of(previous)?;
        Ok(next)
    }

    pub fn clear(previous: &Self) -> Result<Self, String> {
        if previous.merged_into_cluster_id.is_none() {
            return Err("only an active speaker alias can be cleared".to_owned());
        }
        Self::revision_of(previous, None)
    }

    pub fn new_with_id(
        id: Uuid,
        speaker_cluster_id: impl Into<String>,
        parent_revision_id: Option<Uuid>,
        revision: u32,
        merged_into_cluster_id: Option<String>,
    ) -> Result<Self, String> {
        let record = Self {
            id,
            speaker_cluster_id: speaker_cluster_id.into(),
            parent_revision_id,
            revision,
            merged_into_cluster_id,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_revision_id(self.id, "speaker alias revision")?;
        validate_speaker_cluster_id(&self.speaker_cluster_id)?;
        validate_revision_shape(
            self.parent_revision_id,
            self.revision,
            "speaker alias revision",
        )?;
        if self.revision == 1 && self.merged_into_cluster_id.is_none() {
            return Err("the first speaker alias revision must merge into a cluster".to_owned());
        }
        if let Some(merged_into_cluster_id) = &self.merged_into_cluster_id {
            validate_speaker_cluster_id(merged_into_cluster_id)?;
            if merged_into_cluster_id == &self.speaker_cluster_id {
                return Err("a speaker cluster cannot be aliased to itself".to_owned());
            }
        }
        Ok(())
    }

    pub fn validate_for_cluster(&self, cluster: &SpeakerClusterRecord) -> Result<(), String> {
        cluster.validate()?;
        self.validate()?;
        if self.speaker_cluster_id != cluster.id {
            return Err("speaker alias revision must belong to its speaker cluster".to_owned());
        }
        Ok(())
    }

    pub fn validate_successor_of(&self, previous: &Self) -> Result<(), String> {
        previous.validate()?;
        self.validate()?;
        if self.id == previous.id {
            return Err("speaker alias revisions must use distinct IDs".to_owned());
        }
        if self.speaker_cluster_id != previous.speaker_cluster_id {
            return Err("speaker alias revisions must retain their speaker cluster".to_owned());
        }
        if self.parent_revision_id != Some(previous.id) {
            return Err("speaker alias revision must reference its immediate parent".to_owned());
        }
        if self.revision != next_revision_number(previous.revision, "speaker alias")? {
            return Err("speaker alias revision must increase by exactly one".to_owned());
        }
        Ok(())
    }
}

/// Hash-bound payload for atomically creating a cluster and its generated
/// initial label.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerClusterCreatedAuditPayload {
    pub cluster: SpeakerClusterRecord,
    pub initial_label: SpeakerClusterLabelRevision,
}

impl SpeakerClusterCreatedAuditPayload {
    pub fn new(
        cluster: SpeakerClusterRecord,
        initial_label: SpeakerClusterLabelRevision,
    ) -> Result<Self, String> {
        cluster.validate()?;
        initial_label.validate_for_cluster(&cluster)?;
        if initial_label.revision != 1 || initial_label.parent_revision_id.is_some() {
            return Err("initial speaker label must be the first label revision".to_owned());
        }
        if initial_label.is_user_named {
            return Err("initial speaker label must be generated".to_owned());
        }
        Ok(Self {
            cluster,
            initial_label,
        })
    }
}

/// Compact UI projection of an anonymous speaker cluster.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerCluster {
    pub id: String,
    pub session_id: Uuid,
    pub label: String,
    pub is_user_named: bool,
    pub label_revision: u32,
    pub alias_revision: u32,
    pub merged_into_cluster_id: Option<String>,
    pub canonical_cluster_id: String,
    pub span_count: u32,
    pub can_enroll_voice_profile: bool,
}

impl SpeakerCluster {
    pub fn from_revisions(
        record: &SpeakerClusterRecord,
        label: &SpeakerClusterLabelRevision,
        alias: Option<&SpeakerClusterAliasRevision>,
        canonical_cluster_id: impl Into<String>,
        span_count: u32,
        can_enroll_voice_profile: bool,
    ) -> Result<Self, String> {
        record.validate()?;
        label.validate_for_cluster(record)?;
        if let Some(alias) = alias {
            alias.validate_for_cluster(record)?;
        }

        let canonical_cluster_id = canonical_cluster_id.into();
        validate_speaker_cluster_id(&canonical_cluster_id)?;
        let merged_into_cluster_id =
            alias.and_then(|revision| revision.merged_into_cluster_id.clone());
        match &merged_into_cluster_id {
            None if canonical_cluster_id != record.id => {
                return Err(
                    "an active speaker cluster must be its own canonical cluster".to_owned(),
                );
            }
            Some(_) if canonical_cluster_id == record.id => {
                return Err(
                    "an aliased speaker cluster cannot be its own canonical cluster".to_owned(),
                );
            }
            _ => {}
        }

        Ok(Self {
            id: record.id.clone(),
            session_id: record.session_id,
            label: label.label.clone(),
            is_user_named: label.is_user_named,
            label_revision: label.revision,
            alias_revision: alias.map_or(0, |revision| revision.revision),
            merged_into_cluster_id,
            canonical_cluster_id,
            span_count,
            can_enroll_voice_profile,
        })
    }
}

pub fn validate_speaker_cluster_id(value: &str) -> Result<(), String> {
    let Some(uuid) = value.strip_prefix(SPEAKER_CLUSTER_ID_PREFIX) else {
        return Err("speaker cluster ID must use the speaker-<uuid> format".to_owned());
    };
    let parsed = Uuid::parse_str(uuid)
        .map_err(|_| "speaker cluster ID must use the speaker-<uuid> format".to_owned())?;
    if parsed.to_string() != uuid {
        return Err("speaker cluster ID must use a canonical UUID".to_owned());
    }
    Ok(())
}

fn normalize_user_label(value: String) -> Result<String, String> {
    if value.chars().any(char::is_control) {
        return Err("speaker label must not contain control characters".to_owned());
    }
    let normalized = value.trim().to_owned();
    validate_stored_label(&normalized)?;
    Ok(normalized)
}

fn validate_stored_label(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("speaker label must not be empty".to_owned());
    }
    if value != value.trim() {
        return Err("speaker label must not begin or end with whitespace".to_owned());
    }
    if value.chars().any(char::is_control) {
        return Err("speaker label must not contain control characters".to_owned());
    }
    if value.chars().count() > MAX_SPEAKER_LABEL_LENGTH {
        return Err(format!(
            "speaker label must not exceed {MAX_SPEAKER_LABEL_LENGTH} characters"
        ));
    }
    Ok(())
}

fn validate_revision_id(id: Uuid, record_name: &str) -> Result<(), String> {
    if id.is_nil() {
        return Err(format!("{record_name} ID must not be nil"));
    }
    Ok(())
}

fn validate_revision_shape(
    parent_revision_id: Option<Uuid>,
    revision: u32,
    record_name: &str,
) -> Result<(), String> {
    if revision == 0 {
        return Err(format!("{record_name} number must be greater than zero"));
    }
    match (revision, parent_revision_id) {
        (1, None) => Ok(()),
        (1, Some(_)) => Err(format!("initial {record_name} must not have a parent")),
        (_, None) => Err(format!("non-initial {record_name} must have a parent")),
        (_, Some(parent_id)) if parent_id.is_nil() => {
            Err(format!("{record_name} parent ID must not be nil"))
        }
        (_, Some(_)) => Ok(()),
    }
}

fn next_revision_number(previous: u32, record_name: &str) -> Result<u32, String> {
    previous
        .checked_add(1)
        .ok_or_else(|| format!("{record_name} revision number overflowed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_session_scoped_clusters_with_opaque_generated_ids() {
        let session_id = Uuid::new_v4();
        let cluster = SpeakerClusterRecord::new(session_id, 1).unwrap();

        assert_eq!(cluster.session_id, session_id);
        assert_eq!(cluster.ordinal, 1);
        assert!(cluster.id.starts_with(SPEAKER_CLUSTER_ID_PREFIX));
        assert!(validate_speaker_cluster_id(&cluster.id).is_ok());
        assert!(SpeakerClusterRecord::new(session_id, 0).is_err());
        assert!(SpeakerClusterRecord::new_with_id("speaker-legacy", session_id, 1).is_err());
    }

    #[test]
    fn creates_a_generated_initial_label_from_the_anonymous_ordinal() {
        let cluster = SpeakerClusterRecord::new(Uuid::new_v4(), 7).unwrap();
        let label = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let payload =
            SpeakerClusterCreatedAuditPayload::new(cluster.clone(), label.clone()).unwrap();

        assert_eq!(label.speaker_cluster_id, cluster.id);
        assert_eq!(label.parent_revision_id, None);
        assert_eq!(label.revision, 1);
        assert_eq!(label.label, "Speaker 7");
        assert!(!label.is_user_named);
        assert_eq!(payload.initial_label, label);
    }

    #[test]
    fn accepts_bounded_user_labels_and_rejects_invalid_values() {
        let cluster = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let renamed = SpeakerClusterLabelRevision::revision_of(&initial, "  会议主持人  ").unwrap();

        assert_eq!(renamed.label, "会议主持人");
        assert!(renamed.is_user_named);
        assert!(SpeakerClusterLabelRevision::revision_of(&initial, "   ").is_err());
        assert!(SpeakerClusterLabelRevision::revision_of(&initial, "name\nnext").is_err());
        assert!(SpeakerClusterLabelRevision::revision_of(
            &initial,
            "a".repeat(MAX_SPEAKER_LABEL_LENGTH + 1),
        )
        .is_err());
    }

    #[test]
    fn requires_strict_label_revision_increments() {
        let cluster = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let renamed = SpeakerClusterLabelRevision::revision_of(&initial, "主持人").unwrap();

        assert_eq!(renamed.revision, 2);
        assert_eq!(renamed.parent_revision_id, Some(initial.id));

        let skipped = SpeakerClusterLabelRevision::new_with_id(
            Uuid::new_v4(),
            cluster.id,
            Some(initial.id),
            3,
            "主持人",
            true,
        )
        .unwrap();
        assert!(skipped.validate_successor_of(&initial).is_err());
    }

    #[test]
    fn rejects_self_aliases_and_clears_an_alias_by_appending_a_revision() {
        let source = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let target = SpeakerClusterRecord::new(source.session_id, 2).unwrap();

        assert!(
            SpeakerClusterAliasRevision::aliased_to(source.id.clone(), source.id.clone()).is_err()
        );

        let alias = SpeakerClusterAliasRevision::aliased_to(source.id.clone(), target.id).unwrap();
        let cleared = SpeakerClusterAliasRevision::clear(&alias).unwrap();

        assert_eq!(alias.revision, 1);
        assert_eq!(cleared.revision, 2);
        assert_eq!(cleared.parent_revision_id, Some(alias.id));
        assert_eq!(cleared.merged_into_cluster_id, None);
    }

    #[test]
    fn requires_strict_alias_revision_increments() {
        let source = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let target = SpeakerClusterRecord::new(source.session_id, 2).unwrap();
        let alias = SpeakerClusterAliasRevision::aliased_to(source.id.clone(), target.id).unwrap();
        let skipped = SpeakerClusterAliasRevision::new_with_id(
            Uuid::new_v4(),
            source.id,
            Some(alias.id),
            3,
            alias.merged_into_cluster_id.clone(),
        )
        .unwrap();

        assert!(skipped.validate_successor_of(&alias).is_err());
    }

    #[test]
    fn projects_an_active_or_merged_cluster_without_identity_metadata() {
        let source = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let target = SpeakerClusterRecord::new(source.session_id, 2).unwrap();
        let label = SpeakerClusterLabelRevision::initial_generated(&source).unwrap();

        let active =
            SpeakerCluster::from_revisions(&source, &label, None, source.id.clone(), 3, true)
                .unwrap();
        assert_eq!(active.canonical_cluster_id, source.id);
        assert_eq!(active.merged_into_cluster_id, None);
        assert_eq!(active.alias_revision, 0);

        let alias =
            SpeakerClusterAliasRevision::aliased_to(source.id.clone(), target.id.clone()).unwrap();
        let merged =
            SpeakerCluster::from_revisions(&source, &label, Some(&alias), target.id, 3, true)
                .unwrap();
        assert_eq!(
            merged.merged_into_cluster_id,
            merged.canonical_cluster_id.into()
        );
    }
}
