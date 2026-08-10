use super::{AuditEvent, AuditKind, AuditTrail};
use crate::audio::{CaptureGap, CapturePoint};
use crate::domain::{
    CaptureSegment, CaptureSession, SpeakerCluster, SpeakerClusterAliasRevision,
    SpeakerClusterCreatedAuditPayload, SpeakerClusterLabelRevision, SpeakerClusterRecord,
    TranscriptModelProvenance, TranscriptRevision, TranscriptSource,
};
use crate::inference::asr::logical_span_id_for_asr_utterance_digest;
use crate::inference::model_registry::{LocalModelKind, RegisteredModel};
use crate::inference::{
    AsrFinalIdempotencyKey, InferenceGap, InferenceGapReason, InferenceGapStage,
};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};
use uuid::Uuid;

#[derive(Debug)]
pub enum AuditStoreError {
    Database(rusqlite::Error),
    InvalidUuid(String),
    InvalidTimestamp(String),
    InvalidKind(String),
    InvalidCaptureGapReason(String),
    InvalidCaptureMetadata {
        field: &'static str,
        value: String,
    },
    InvalidCaptureGapRange,
    InvalidInferenceGapMetadata {
        field: &'static str,
        value: String,
    },
    InvalidTranscriptMetadata {
        field: &'static str,
        value: String,
    },
    InvalidTranscriptRange,
    InvalidTranscriptWallClockRange,
    NonFinalTranscript,
    MissingTranscriptParent(String),
    InvalidTranscriptParent {
        parent_id: String,
        reason: &'static str,
    },
    InvalidModelKind(String),
    InvalidModelMetadata {
        field: &'static str,
        value: String,
    },
    InvalidSpeakerMetadata {
        field: &'static str,
        value: String,
    },
    Integrity,
}

impl std::fmt::Display for AuditStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(error) => write!(formatter, "database error: {error}"),
            Self::InvalidUuid(value) => write!(formatter, "invalid UUID in audit store: {value}"),
            Self::InvalidTimestamp(value) => {
                write!(formatter, "invalid timestamp in audit store: {value}")
            }
            Self::InvalidKind(value) => {
                write!(formatter, "invalid audit kind in audit store: {value}")
            }
            Self::InvalidCaptureGapReason(value) => {
                write!(
                    formatter,
                    "invalid capture gap reason in audit store: {value}"
                )
            }
            Self::InvalidCaptureMetadata { field, value } => {
                write!(formatter, "invalid capture metadata for {field}: {value}")
            }
            Self::InvalidCaptureGapRange => {
                formatter.write_str("capture gap end must not precede its start")
            }
            Self::InvalidInferenceGapMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid inference gap metadata for {field}: {value}"
                )
            }
            Self::InvalidTranscriptMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid transcript metadata for {field}: {value}"
                )
            }
            Self::InvalidTranscriptRange => {
                formatter.write_str("transcript capture end must not precede its start")
            }
            Self::InvalidTranscriptWallClockRange => {
                formatter.write_str("transcript wall-clock end must not precede its start")
            }
            Self::NonFinalTranscript => {
                formatter.write_str("only final transcript revisions may be persisted")
            }
            Self::MissingTranscriptParent(parent_id) => {
                write!(
                    formatter,
                    "transcript revision parent does not exist: {parent_id}"
                )
            }
            Self::InvalidTranscriptParent { parent_id, reason } => {
                write!(
                    formatter,
                    "invalid transcript revision parent {parent_id}: {reason}"
                )
            }
            Self::InvalidModelKind(value) => {
                write!(
                    formatter,
                    "invalid local model kind in audit store: {value}"
                )
            }
            Self::InvalidModelMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid local model metadata for {field}: {value}"
                )
            }
            Self::InvalidSpeakerMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid speaker catalog metadata for {field}: {value}"
                )
            }
            Self::Integrity => write!(formatter, "audit chain integrity check failed"),
        }
    }
}

impl std::error::Error for AuditStoreError {}

impl From<rusqlite::Error> for AuditStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

pub struct AuditStore {
    connection: Connection,
}

/// Durable result of one native ASR final emission. This is deliberately
/// separate from `transcript_revisions`: final-ASR revisions may have an
/// adapter revision that differs from the first durable transcript revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AsrFinalIdempotencyRecord {
    pub revision_id: Uuid,
    pub emission_payload_sha256: String,
}

/// One immutable inference-gap record together with the exact audit event
/// that commits it. Callers use this to make a lost post-commit acknowledgement
/// replayable without extending the hash chain a second time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceGapAuditRecord {
    pub gap: InferenceGap,
    pub audit_event: AuditEvent,
}

/// The immutable facts that bind a native ASR final emission to its durable
/// transcript revision. This object is included in the transcript audit
/// event's hashed payload, so changing SQLite idempotency metadata is
/// detectable during verification.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrFinalIdempotencyBinding {
    pub session_id: Uuid,
    pub utterance_key_sha256: String,
    pub emission_revision: u32,
    pub revision_id: Uuid,
    pub logical_span_id: Uuid,
    pub emission_payload_sha256: String,
}

impl AsrFinalIdempotencyBinding {
    pub fn new(
        key: &AsrFinalIdempotencyKey,
        revision: &TranscriptRevision,
        emission_payload_sha256: impl Into<String>,
    ) -> Result<Self, String> {
        key.validate()?;
        let emission_payload_sha256 = emission_payload_sha256.into();
        if emission_payload_sha256.len() != 64
            || !emission_payload_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("ASR final payload SHA-256 must be 64 hexadecimal characters".to_owned());
        }
        let utterance_key_sha256 = key.opaque_utterance_key_sha256();
        let expected_logical_span_id =
            logical_span_id_for_asr_utterance_digest(key.session_id, &utterance_key_sha256);
        if revision.logical_span_id != expected_logical_span_id {
            return Err("ASR final logical span ID does not match its utterance key".to_owned());
        }
        if revision.id != revision.logical_span_id {
            return Err("ASR final must bind its first durable transcript revision".to_owned());
        }

        Ok(Self {
            session_id: key.session_id,
            utterance_key_sha256,
            emission_revision: key.emission_revision,
            revision_id: revision.id,
            logical_span_id: revision.logical_span_id,
            emission_payload_sha256,
        })
    }
}

/// Payload committed by the existing transcript-revision audit event when
/// the revision originated from native ASR. It preserves the standard
/// transcript event kind while binding ASR replay metadata into its digest.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrFinalAuditPayload<'a> {
    pub revision: &'a TranscriptRevision,
    pub idempotency: &'a AsrFinalIdempotencyBinding,
}

impl<'a> AsrFinalAuditPayload<'a> {
    pub fn new(
        revision: &'a TranscriptRevision,
        idempotency: &'a AsrFinalIdempotencyBinding,
    ) -> Self {
        Self {
            revision,
            idempotency,
        }
    }
}

impl AuditStore {
    pub fn open_in_memory() -> Result<Self, AuditStoreError> {
        let connection = Connection::open_in_memory()?;
        Self::from_connection(connection)
    }

    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, AuditStoreError> {
        let connection = Connection::open(path)?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> Result<Self, AuditStoreError> {
        connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                run_id TEXT,
                causation_id TEXT,
                kind TEXT NOT NULL,
                monotonic_ns TEXT NOT NULL,
                wall_clock TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                previous_hash TEXT,
                hash TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS capture_segments (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                device_uid TEXT NOT NULL,
                device_name TEXT NOT NULL,
                sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
                channels INTEGER NOT NULL CHECK (channels > 0),
                anchor_monotonic_ns TEXT NOT NULL,
                anchor_wall_clock TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS capture_segments_session_sequence
                ON capture_segments(session_id, sequence);

            CREATE TABLE IF NOT EXISTS capture_gaps (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                started_monotonic_ns TEXT NOT NULL,
                started_wall_clock TEXT NOT NULL,
                ended_monotonic_ns TEXT NOT NULL,
                ended_wall_clock TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS capture_gaps_session_sequence
                ON capture_gaps(session_id, sequence);

            CREATE TABLE IF NOT EXISTS inference_gaps (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                runtime_id TEXT NOT NULL,
                capture_segment_id TEXT NOT NULL,
                job_id TEXT,
                started_monotonic_ns TEXT NOT NULL,
                started_wall_clock TEXT NOT NULL,
                ended_monotonic_ns TEXT NOT NULL,
                ended_wall_clock TEXT NOT NULL,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                audit_event_id TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS inference_gaps_session_sequence
                ON inference_gaps(session_id, sequence);
            CREATE INDEX IF NOT EXISTS inference_gaps_runtime_sequence
                ON inference_gaps(runtime_id, sequence);
            CREATE TRIGGER IF NOT EXISTS inference_gaps_are_immutable_update
            BEFORE UPDATE ON inference_gaps
            BEGIN
                SELECT RAISE(ABORT, 'inference gaps are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS inference_gaps_are_immutable_delete
            BEFORE DELETE ON inference_gaps
            BEGIN
                SELECT RAISE(ABORT, 'inference gaps are immutable');
            END;

            CREATE TABLE IF NOT EXISTS transcript_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                logical_span_id TEXT NOT NULL,
                parent_revision_id TEXT,
                session_id TEXT NOT NULL,
                capture_start_ns TEXT NOT NULL,
                capture_end_ns TEXT NOT NULL,
                wall_clock_start TEXT NOT NULL,
                wall_clock_end TEXT NOT NULL,
                speaker_cluster_id TEXT,
                text TEXT NOT NULL,
                is_final INTEGER NOT NULL CHECK (is_final IN (0, 1)),
                revision INTEGER NOT NULL CHECK (revision >= 0),
                source TEXT NOT NULL,
                model_provider TEXT,
                model_id TEXT,
                model_version TEXT,
                model_sha256 TEXT,
                confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0))
            );
            CREATE INDEX IF NOT EXISTS transcript_revisions_session_sequence
                ON transcript_revisions(session_id, sequence);
            CREATE UNIQUE INDEX IF NOT EXISTS transcript_revisions_logical_revision
                ON transcript_revisions(logical_span_id, revision);
            CREATE INDEX IF NOT EXISTS transcript_revisions_parent
                ON transcript_revisions(parent_revision_id);

            CREATE TRIGGER IF NOT EXISTS transcript_revisions_are_immutable_update
            BEFORE UPDATE ON transcript_revisions
            BEGIN
                SELECT RAISE(ABORT, 'transcript revisions are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS transcript_revisions_are_immutable_delete
            BEFORE DELETE ON transcript_revisions
            BEGIN
                SELECT RAISE(ABORT, 'transcript revisions are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_clusters (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal > 0),
                audit_event_id TEXT NOT NULL UNIQUE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS speaker_clusters_session_ordinal
                ON speaker_clusters(session_id, ordinal);
            CREATE INDEX IF NOT EXISTS speaker_clusters_session_sequence
                ON speaker_clusters(session_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_clusters_are_immutable_update
            BEFORE UPDATE ON speaker_clusters
            BEGIN
                SELECT RAISE(ABORT, 'speaker clusters are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_clusters_are_immutable_delete
            BEFORE DELETE ON speaker_clusters
            BEGIN
                SELECT RAISE(ABORT, 'speaker clusters are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_cluster_label_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                speaker_cluster_id TEXT NOT NULL,
                parent_revision_id TEXT,
                revision INTEGER NOT NULL CHECK (revision > 0),
                label TEXT NOT NULL,
                is_user_named INTEGER NOT NULL CHECK (is_user_named IN (0, 1)),
                audit_event_id TEXT NOT NULL UNIQUE,
                UNIQUE(speaker_cluster_id, revision)
            );
            CREATE INDEX IF NOT EXISTS speaker_cluster_label_revisions_cluster_sequence
                ON speaker_cluster_label_revisions(speaker_cluster_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_cluster_label_revisions_are_immutable_update
            BEFORE UPDATE ON speaker_cluster_label_revisions
            BEGIN
                SELECT RAISE(ABORT, 'speaker label revisions are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_cluster_label_revisions_are_immutable_delete
            BEFORE DELETE ON speaker_cluster_label_revisions
            BEGIN
                SELECT RAISE(ABORT, 'speaker label revisions are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_cluster_alias_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                speaker_cluster_id TEXT NOT NULL,
                parent_revision_id TEXT,
                revision INTEGER NOT NULL CHECK (revision > 0),
                merged_into_cluster_id TEXT,
                audit_event_id TEXT NOT NULL UNIQUE,
                UNIQUE(speaker_cluster_id, revision)
            );
            CREATE INDEX IF NOT EXISTS speaker_cluster_alias_revisions_cluster_sequence
                ON speaker_cluster_alias_revisions(speaker_cluster_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_cluster_alias_revisions_are_immutable_update
            BEFORE UPDATE ON speaker_cluster_alias_revisions
            BEGIN
                SELECT RAISE(ABORT, 'speaker alias revisions are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_cluster_alias_revisions_are_immutable_delete
            BEFORE DELETE ON speaker_cluster_alias_revisions
            BEGIN
                SELECT RAISE(ABORT, 'speaker alias revisions are immutable');
            END;

            CREATE TABLE IF NOT EXISTS asr_final_idempotency (
                session_id TEXT NOT NULL,
                utterance_key_sha256 TEXT NOT NULL,
                emission_revision INTEGER NOT NULL CHECK (emission_revision > 0),
                revision_id TEXT NOT NULL UNIQUE,
                emission_payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (session_id, utterance_key_sha256, emission_revision)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS asr_final_idempotency_utterance
                ON asr_final_idempotency(session_id, utterance_key_sha256);
            CREATE TRIGGER IF NOT EXISTS asr_final_idempotency_is_immutable_update
            BEFORE UPDATE ON asr_final_idempotency
            BEGIN
                SELECT RAISE(ABORT, 'ASR final idempotency records are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS asr_final_idempotency_is_immutable_delete
            BEFORE DELETE ON asr_final_idempotency
            BEGIN
                SELECT RAISE(ABORT, 'ASR final idempotency records are immutable');
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS transcript_revision_fts USING fts5(
                text,
                revision_id UNINDEXED,
                session_id UNINDEXED,
                logical_span_id UNINDEXED,
                tokenize = 'unicode61'
            );

            CREATE TABLE IF NOT EXISTS local_models (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                model_kind TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                file_size_bytes TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                version TEXT NOT NULL,
                input_format TEXT NOT NULL,
                model_card_id TEXT NOT NULL,
                license_id TEXT NOT NULL,
                license_confirmed_at TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS local_models_kind_imported
                ON local_models(model_kind, imported_at);
            ",
        )?;
        Ok(Self { connection })
    }

    pub fn append(&self, event: &AuditEvent) -> Result<(), AuditStoreError> {
        insert_audit_event(&self.connection, event)
    }

    /// Atomically creates an anonymous speaker cluster and its generated
    /// initial label. The single audit event hashes both immutable records.
    pub fn append_speaker_cluster_with_audit(
        &mut self,
        event: &AuditEvent,
        cluster: &SpeakerClusterRecord,
        initial_label: &SpeakerClusterLabelRevision,
    ) -> Result<(), AuditStoreError> {
        validate_speaker_cluster_created_audit_event(event, cluster, initial_label)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_speaker_cluster(&transaction, cluster, event.id)?;
        insert_speaker_cluster_label_revision(&transaction, initial_label, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    /// Appends one user-facing display-label correction without overwriting
    /// the generated label or an earlier correction.
    pub fn append_speaker_cluster_label_revision_with_audit(
        &mut self,
        event: &AuditEvent,
        revision: &SpeakerClusterLabelRevision,
    ) -> Result<(), AuditStoreError> {
        let cluster =
            validate_speaker_cluster_label_revision_for_write(&self.connection, revision)?;
        validate_speaker_cluster_label_revision_audit_event(event, &cluster, revision)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_speaker_cluster_label_revision(&transaction, revision, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    /// Appends a reversible alias (or an explicit alias clearing) without
    /// changing historical transcript assignments.
    pub fn append_speaker_cluster_alias_revision_with_audit(
        &mut self,
        event: &AuditEvent,
        revision: &SpeakerClusterAliasRevision,
    ) -> Result<(), AuditStoreError> {
        let cluster =
            validate_speaker_cluster_alias_revision_for_write(&self.connection, revision)?;
        validate_speaker_cluster_alias_revision_audit_event(event, &cluster, revision)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_speaker_cluster_alias_revision(&transaction, revision, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    /// Returns compact current catalog projections for one session. Old
    /// transcript values such as `speaker-1` do not need catalog rows and are
    /// intentionally excluded from this list.
    pub fn list_speaker_clusters(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<SpeakerCluster>, AuditStoreError> {
        let catalog = load_speaker_catalog(&self.connection)?;
        let aliases = catalog.active_aliases();
        let mut records = catalog
            .clusters
            .values()
            .filter(|stored| stored.record.session_id == session_id)
            .collect::<Vec<_>>();
        records.sort_by_key(|stored| stored.record.ordinal);
        records
            .into_iter()
            .map(|stored| {
                let label = catalog.latest_label(&stored.record.id).ok_or_else(|| {
                    speaker_error("speaker label", "cluster has no label revision")
                })?;
                let alias = catalog.latest_alias(&stored.record.id);
                let canonical_cluster_id = resolve_speaker_cluster_canonical_id(
                    &stored.record.id,
                    &catalog.clusters,
                    &aliases,
                )?;
                let span_count = current_speaker_cluster_span_count(
                    &self.connection,
                    session_id,
                    &stored.record.id,
                )?;
                let span_count = span_count
                    .try_into()
                    .map_err(|_| speaker_error("speaker span count", span_count.to_string()))?;
                SpeakerCluster::from_revisions(
                    &stored.record,
                    &label.revision,
                    alias.map(|stored| &stored.revision),
                    canonical_cluster_id,
                    span_count,
                )
                .map_err(|value| speaker_error("speaker projection", value))
            })
            .collect()
    }

    /// Returns the immutable catalog record for a known opaque cluster ID.
    /// This does not interpret legacy transcript strings as catalog records.
    pub fn get_speaker_cluster_record(
        &self,
        cluster_id: &str,
    ) -> Result<Option<SpeakerClusterRecord>, AuditStoreError> {
        let catalog = load_speaker_catalog(&self.connection)?;
        Ok(catalog
            .clusters
            .get(cluster_id)
            .map(|stored| stored.record.clone()))
    }

    /// Returns the current immutable display-label revision needed to append
    /// an optimistic-concurrency-safe user correction.
    pub fn get_latest_speaker_cluster_label_revision(
        &self,
        cluster_id: &str,
    ) -> Result<Option<SpeakerClusterLabelRevision>, AuditStoreError> {
        let catalog = load_speaker_catalog(&self.connection)?;
        Ok(catalog
            .latest_label(cluster_id)
            .map(|stored| stored.revision.clone()))
    }

    /// Atomically persist the three audit records that publish a new native
    /// capture session. The caller updates its in-memory session projection
    /// only after this bundle commits, so a staged microphone failure cannot
    /// leave a standalone `SessionStarted` record behind.
    pub fn append_capture_start_bundle_with_audit<T: Serialize>(
        &mut self,
        session: &CaptureSession,
        segment: &CaptureSegment,
        session_started: &AuditEvent,
        segment_recorded: &AuditEvent,
        input_started: &AuditEvent,
        input_started_payload: &T,
    ) -> Result<(), AuditStoreError> {
        validate_capture_start_bundle(
            &self.connection,
            session,
            segment,
            session_started,
            segment_recorded,
            input_started,
            input_started_payload,
        )?;

        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, session_started)?;
        insert_audit_event(&transaction, segment_recorded)?;
        insert_capture_segment(&transaction, segment)?;
        insert_audit_event(&transaction, input_started)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<AuditEvent>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, run_id, causation_id, kind, monotonic_ns, wall_clock, payload_hash, previous_hash, hash
            FROM audit_events
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
            ))
        })?;

        rows.map(|row| {
            let (
                id,
                run_id,
                causation_id,
                kind,
                monotonic_ns,
                wall_clock,
                payload_hash,
                previous_hash,
                hash,
            ) = row?;
            Ok(AuditEvent {
                id: parse_uuid(&id)?,
                run_id: parse_optional_uuid(run_id)?,
                causation_id: parse_optional_uuid(causation_id)?,
                kind: serde_json::from_str(&kind)
                    .map_err(|_| AuditStoreError::InvalidKind(kind))?,
                monotonic_ns: monotonic_ns
                    .parse()
                    .map_err(|_| AuditStoreError::InvalidTimestamp(monotonic_ns))?,
                wall_clock: parse_timestamp(&wall_clock)?,
                payload_hash,
                previous_hash,
                hash,
            })
        })
        .collect()
    }

    pub fn verify(&self) -> Result<bool, AuditStoreError> {
        let events = self.list()?;
        if !AuditTrail::from_events(events.clone()).verify() {
            return Ok(false);
        }

        // The hash chain proves event ordering; these checks additionally
        // prove that every durable M2 record still matches exactly one event
        // after reopening the SQLite database.
        let transcript_events = events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    AuditKind::TranscriptRevisionRecorded | AuditKind::TranscriptSpeakerReassigned
                )
            })
            .collect::<Vec<_>>();
        let revisions = self.list_all_transcript_revisions()?;
        let revisions_by_id = revisions
            .iter()
            .map(|revision| (revision.id, revision))
            .collect::<BTreeMap<_, _>>();
        let Some(asr_bindings) = self.verified_asr_final_idempotency_bindings()? else {
            return Ok(false);
        };
        if transcript_events.len() != revisions.len()
            || !revisions.iter().all(|revision| {
                let matches_event = |event: &&AuditEvent| match asr_bindings.get(&revision.id) {
                    Some(binding) => {
                        validate_asr_final_audit_event(event, revision, binding).is_ok()
                    }
                    None if event.kind == AuditKind::TranscriptSpeakerReassigned => revision
                        .parent_revision_id
                        .and_then(|parent_id| revisions_by_id.get(&parent_id).copied())
                        .is_some_and(|parent| {
                            validate_transcript_speaker_reassignment_audit_event(
                                &self.connection,
                                event,
                                revision,
                                parent,
                            )
                            .is_ok()
                        }),
                    None => {
                        let is_speaker_only = revision
                            .parent_revision_id
                            .and_then(|parent_id| revisions_by_id.get(&parent_id).copied())
                            .is_some_and(|parent| is_speaker_only_reassignment(revision, parent));
                        !is_speaker_only && validate_transcript_audit_event(event, revision).is_ok()
                    }
                };
                transcript_events.iter().any(matches_event)
            })
        {
            return Ok(false);
        }

        let model_events = events
            .iter()
            .filter(|event| event.kind == AuditKind::LocalModelImported)
            .collect::<Vec<_>>();
        let models = self.list_local_models()?;
        if model_events.len() != models.len()
            || !models.iter().all(|model| {
                model_events
                    .iter()
                    .any(|event| validate_local_model_audit_event(event, model).is_ok())
            })
        {
            return Ok(false);
        }

        let inference_events = events
            .iter()
            .filter(|event| event.kind == AuditKind::InferenceGapRecorded)
            .map(|event| (event.id, event))
            .collect::<BTreeMap<_, _>>();
        let inference_gaps = self.list_all_inference_gap_records()?;
        if inference_events.len() != inference_gaps.len() {
            return Ok(false);
        }
        let mut bound_events = BTreeSet::new();
        for record in inference_gaps {
            let Some(event) = inference_events.get(&record.audit_event_id) else {
                return Ok(false);
            };
            if !bound_events.insert(record.audit_event_id)
                || validate_inference_gap_audit_event(event, &record.gap).is_err()
            {
                return Ok(false);
            }
        }

        let catalog = match load_speaker_catalog(&self.connection) {
            Ok(catalog) => catalog,
            Err(_) => return Ok(false),
        };
        if !verify_speaker_catalog(&events, &catalog) {
            return Ok(false);
        }

        Ok(true)
    }

    /// Persist metadata for a stream lifetime without retaining PCM samples.
    pub fn append_capture_segment(&self, segment: &CaptureSegment) -> Result<(), AuditStoreError> {
        validate_capture_segment(segment)?;
        insert_capture_segment(&self.connection, segment)
    }

    /// Write a segment and its hash-chain event in one SQLite transaction.
    pub fn append_capture_segment_with_audit(
        &mut self,
        event: &AuditEvent,
        segment: &CaptureSegment,
    ) -> Result<(), AuditStoreError> {
        validate_capture_segment(segment)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_capture_segment(&transaction, segment)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_capture_segments(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<CaptureSegment>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, session_id, device_uid, device_name, sample_rate, channels,
                   anchor_monotonic_ns, anchor_wall_clock
            FROM capture_segments
            WHERE session_id = ?1
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map(params![session_id.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;

        rows.map(|row| {
            let (
                id,
                stored_session_id,
                device_uid,
                device_name,
                sample_rate,
                channels,
                anchor_monotonic_ns,
                anchor_wall_clock,
            ) = row?;
            Ok(CaptureSegment {
                id: parse_uuid(&id)?,
                session_id: parse_uuid(&stored_session_id)?,
                device_uid,
                device_name,
                sample_rate: parse_capture_integer("sample rate", sample_rate)?,
                channels: parse_capture_integer("channel count", channels)?,
                anchor_monotonic_ns: parse_capture_monotonic_ns(&anchor_monotonic_ns)?,
                anchor_wall_clock: parse_timestamp(&anchor_wall_clock)?,
            })
        })
        .collect()
    }

    /// Persist a capture discontinuity without retaining audio data.
    pub fn append_capture_gap(
        &self,
        session_id: Uuid,
        gap: &CaptureGap,
    ) -> Result<(), AuditStoreError> {
        validate_capture_gap(gap)?;
        insert_capture_gap(&self.connection, session_id, gap)
    }

    /// Write a capture discontinuity and its hash-chain event atomically.
    pub fn append_capture_gap_with_audit(
        &mut self,
        event: &AuditEvent,
        session_id: Uuid,
        gap: &CaptureGap,
    ) -> Result<(), AuditStoreError> {
        validate_capture_gap(gap)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_capture_gap(&transaction, session_id, gap)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_capture_gaps(&self, session_id: Uuid) -> Result<Vec<CaptureGap>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT started_monotonic_ns, started_wall_clock,
                   ended_monotonic_ns, ended_wall_clock, reason
            FROM capture_gaps
            WHERE session_id = ?1
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map(params![session_id.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;

        rows.map(|row| {
            let (
                started_monotonic_ns,
                started_wall_clock,
                ended_monotonic_ns,
                ended_wall_clock,
                reason,
            ) = row?;
            let started_at = CapturePoint {
                monotonic_ns: parse_capture_monotonic_ns(&started_monotonic_ns)?,
                wall_clock: parse_timestamp(&started_wall_clock)?,
            };
            let ended_at = CapturePoint {
                monotonic_ns: parse_capture_monotonic_ns(&ended_monotonic_ns)?,
                wall_clock: parse_timestamp(&ended_wall_clock)?,
            };
            if ended_at.monotonic_ns < started_at.monotonic_ns {
                return Err(AuditStoreError::InvalidCaptureGapRange);
            }
            Ok(CaptureGap {
                started_at,
                ended_at,
                reason: serde_json::from_str(&reason)
                    .map_err(|_| AuditStoreError::InvalidCaptureGapReason(reason))?,
            })
        })
        .collect()
    }

    /// Write an inference terminal outcome and its audit event atomically.
    /// The record is intentionally separate from capture gaps because the
    /// audio range was captured but did not yield a final transcript.
    pub fn append_inference_gap_with_audit(
        &mut self,
        event: &AuditEvent,
        gap: &InferenceGap,
    ) -> Result<(), AuditStoreError> {
        validate_inference_gap_audit_event(event, gap)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_inference_gap(&transaction, gap, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_inference_gaps(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<InferenceGap>, AuditStoreError> {
        self.list_inference_gap_records(Some(session_id))
            .map(|records| records.into_iter().map(|record| record.gap).collect())
    }

    /// Returns the durable inference gap and the audit event bound to it.
    ///
    /// The relationship is revalidated on read because this API is used to
    /// decide whether a worker outcome can be acknowledged after a retry.
    pub fn lookup_inference_gap_with_audit(
        &self,
        gap_id: Uuid,
    ) -> Result<Option<InferenceGapAuditRecord>, AuditStoreError> {
        let stored = self
            .connection
            .query_row(
                "
                SELECT id, session_id, runtime_id, capture_segment_id, job_id,
                       started_monotonic_ns, started_wall_clock,
                       ended_monotonic_ns, ended_wall_clock, stage, reason, audit_event_id
                FROM inference_gaps
                WHERE id = ?1
                ",
                params![gap_id.to_string()],
                inference_gap_row,
            )
            .optional()?;
        let Some(stored) = stored else {
            return Ok(None);
        };
        let record = parse_inference_gap_record(stored)?;
        let event = self
            .list()?
            .into_iter()
            .find(|event| event.id == record.audit_event_id)
            .ok_or(AuditStoreError::Integrity)?;
        if !event.verifies() {
            return Err(AuditStoreError::Integrity);
        }
        validate_inference_gap_audit_event(&event, &record.gap)?;

        Ok(Some(InferenceGapAuditRecord {
            gap: record.gap,
            audit_event: event,
        }))
    }

    /// Append a transcript revision and its audit event in one SQLite
    /// transaction, so a persisted revision never exists without its audit
    /// record and vice versa.
    pub fn append_transcript_revision_with_audit(
        &mut self,
        event: &AuditEvent,
        revision: &TranscriptRevision,
    ) -> Result<(), AuditStoreError> {
        validate_transcript_audit_event(event, revision)?;
        reject_speaker_reassignment_from_generic_path(&self.connection, revision)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_transcript_revision(&transaction, revision)?;
        transaction.commit()?;
        Ok(())
    }

    /// Appends a speaker-only correction as a dedicated audit action. Its
    /// event is timestamped when the user makes the correction, while the
    /// immutable transcript revision retains its original capture timing.
    pub fn append_transcript_speaker_reassignment_with_audit(
        &mut self,
        event: &AuditEvent,
        revision: &TranscriptRevision,
    ) -> Result<(), AuditStoreError> {
        validate_transcript_speaker_reassignment_with_audit(&self.connection, event, revision)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_transcript_revision(&transaction, revision)?;
        transaction.commit()?;
        Ok(())
    }

    /// Returns the durable final-ASR record for one source emission identity.
    /// Callers compare its payload digest before treating a replay as benign.
    pub fn lookup_asr_final_idempotency(
        &self,
        key: &AsrFinalIdempotencyKey,
    ) -> Result<Option<AsrFinalIdempotencyRecord>, AuditStoreError> {
        validate_asr_final_idempotency_key(key)?;
        self.connection
            .query_row(
                "
                SELECT revision_id, emission_payload_sha256
                FROM asr_final_idempotency
                WHERE session_id = ?1 AND utterance_key_sha256 = ?2 AND emission_revision = ?3
                ",
                params![
                    key.session_id.to_string(),
                    key.opaque_utterance_key_sha256(),
                    i64::from(key.emission_revision),
                ],
                |row| {
                    Ok(AsrFinalIdempotencyRecord {
                        revision_id: parse_uuid(&row.get::<_, String>(0)?).map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                0,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })?,
                        emission_payload_sha256: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(AuditStoreError::from)
    }

    /// Writes an audit event, its immutable transcript revision, and the
    /// native final-ASR idempotency key in one transaction.
    pub fn append_asr_final_transcript_revision_with_audit(
        &mut self,
        event: &AuditEvent,
        revision: &TranscriptRevision,
        idempotency: &AsrFinalIdempotencyBinding,
    ) -> Result<(), AuditStoreError> {
        validate_asr_final_audit_event(event, revision, idempotency)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_transcript_revision(&transaction, revision)?;
        insert_asr_final_idempotency(&transaction, idempotency, revision.id)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_transcript_revisions(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<TranscriptRevision>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, logical_span_id, parent_revision_id, session_id,
                   capture_start_ns, capture_end_ns, wall_clock_start, wall_clock_end,
                   speaker_cluster_id, text, is_final, revision, source,
                   model_provider, model_id, model_version, model_sha256, confidence
            FROM transcript_revisions
            WHERE session_id = ?1
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map(params![session_id.to_string()], transcript_revision_row)?;
        rows.map(|row| parse_transcript_revision(row?)).collect()
    }

    /// Lists all locally persisted transcript revisions in insertion order.
    ///
    /// This is used only to rebuild the in-process timeline projection when
    /// WordCovenant opens its own SQLite database; it never exposes PCM.
    pub fn list_all_transcript_revisions(
        &self,
    ) -> Result<Vec<TranscriptRevision>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, logical_span_id, parent_revision_id, session_id,
                   capture_start_ns, capture_end_ns, wall_clock_start, wall_clock_end,
                   speaker_cluster_id, text, is_final, revision, source,
                   model_provider, model_id, model_version, model_sha256, confidence
            FROM transcript_revisions
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map([], transcript_revision_row)?;
        rows.map(|row| parse_transcript_revision(row?)).collect()
    }

    /// Atomically persists an application-managed model registration and its
    /// audit event. The model bytes have already been copied locally by the
    /// registry before this method is called.
    pub fn append_local_model_with_audit(
        &mut self,
        event: &AuditEvent,
        model: &RegisteredModel,
    ) -> Result<(), AuditStoreError> {
        validate_local_model_audit_event(event, model)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_registered_model(&transaction, model)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_local_models(&self) -> Result<Vec<RegisteredModel>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, model_kind, file_path, file_size_bytes, sha256, version,
                   input_format, model_card_id, license_id, license_confirmed_at, imported_at
            FROM local_models
            ORDER BY sequence ASC
            ",
        )?;
        let rows = statement.query_map([], local_model_row)?;
        rows.map(|row| parse_registered_model(row?)).collect()
    }

    /// Searches only the on-device FTS5 projection. Search returns immutable
    /// revisions rather than silently substituting a newer transcript value.
    pub fn search_transcript_revisions(
        &self,
        session_id: Option<Uuid>,
        query: &str,
    ) -> Result<Vec<TranscriptRevision>, AuditStoreError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut statement = self.connection.prepare(
            "
            SELECT revisions.id, revisions.logical_span_id, revisions.parent_revision_id,
                   revisions.session_id, revisions.capture_start_ns, revisions.capture_end_ns,
                   revisions.wall_clock_start, revisions.wall_clock_end,
                   revisions.speaker_cluster_id, revisions.text, revisions.is_final,
                   revisions.revision, revisions.source, revisions.model_provider,
                   revisions.model_id, revisions.model_version, revisions.model_sha256,
                   revisions.confidence
            FROM transcript_revision_fts
            INNER JOIN transcript_revisions AS revisions
                ON revisions.id = transcript_revision_fts.revision_id
            WHERE transcript_revision_fts MATCH ?1
              AND (?2 IS NULL OR revisions.session_id = ?2)
            ORDER BY revisions.wall_clock_start ASC, revisions.sequence ASC
            ",
        )?;
        let stored_session_id = session_id.map(|value| value.to_string());
        let rows =
            statement.query_map(params![query, stored_session_id], transcript_revision_row)?;
        rows.map(|row| parse_transcript_revision(row?)).collect()
    }

    fn list_all_inference_gap_records(&self) -> Result<Vec<InferenceGapRecord>, AuditStoreError> {
        self.list_inference_gap_records(None)
    }

    fn list_inference_gap_records(
        &self,
        session_id: Option<Uuid>,
    ) -> Result<Vec<InferenceGapRecord>, AuditStoreError> {
        let mut statement = self.connection.prepare(
            "
            SELECT id, session_id, runtime_id, capture_segment_id, job_id,
                   started_monotonic_ns, started_wall_clock,
                   ended_monotonic_ns, ended_wall_clock, stage, reason, audit_event_id
            FROM inference_gaps
            WHERE (?1 IS NULL OR session_id = ?1)
            ORDER BY sequence ASC
            ",
        )?;
        let stored_session_id = session_id.map(|value| value.to_string());
        let rows = statement.query_map(params![stored_session_id], inference_gap_row)?;
        rows.map(|row| parse_inference_gap_record(row?)).collect()
    }

    fn verified_asr_final_idempotency_bindings(
        &self,
    ) -> Result<Option<BTreeMap<Uuid, AsrFinalIdempotencyBinding>>, AuditStoreError> {
        let broken_references = self.connection.query_row(
            "
            SELECT COUNT(*)
            FROM asr_final_idempotency AS keys
            LEFT JOIN transcript_revisions AS revisions
                ON revisions.id = keys.revision_id
            WHERE revisions.id IS NULL
               OR revisions.session_id != keys.session_id
               OR revisions.is_final != 1
               OR revisions.source != ?1
            ",
            params![serde_json::to_string(&TranscriptSource::LocalInference)
                .expect("transcript source serializes")],
            |row| row.get::<_, i64>(0),
        )?;
        if broken_references != 0 {
            return Ok(None);
        }

        let mut statement = self.connection.prepare(
            "
            SELECT keys.session_id, keys.utterance_key_sha256, keys.emission_revision,
                   keys.revision_id, keys.emission_payload_sha256,
                   revisions.logical_span_id
            FROM asr_final_idempotency AS keys
            INNER JOIN transcript_revisions AS revisions
                ON revisions.id = keys.revision_id
            ",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;
        let mut bindings = BTreeMap::new();
        for row in rows {
            let (
                session_id,
                utterance_key_sha256,
                emission_revision,
                revision_id,
                emission_payload_sha256,
                logical_span_id,
            ) = row?;
            let emission_revision = emission_revision.try_into().map_err(|_| {
                AuditStoreError::InvalidTranscriptMetadata {
                    field: "ASR final idempotency revision",
                    value: emission_revision.to_string(),
                }
            })?;
            let binding = AsrFinalIdempotencyBinding {
                session_id: parse_uuid(&session_id)?,
                utterance_key_sha256,
                emission_revision,
                revision_id: parse_uuid(&revision_id)?,
                logical_span_id: parse_uuid(&logical_span_id)?,
                emission_payload_sha256,
            };
            if validate_persisted_asr_final_idempotency_binding(&binding).is_err()
                || bindings.insert(binding.revision_id, binding).is_some()
            {
                return Ok(None);
            }
        }

        Ok(Some(bindings))
    }
}

struct TranscriptRevisionRow {
    id: String,
    logical_span_id: String,
    parent_revision_id: Option<String>,
    session_id: String,
    capture_start_ns: String,
    capture_end_ns: String,
    wall_clock_start: String,
    wall_clock_end: String,
    speaker_cluster_id: Option<String>,
    text: String,
    is_final: i64,
    revision: i64,
    source: String,
    model_provider: Option<String>,
    model_id: Option<String>,
    model_version: Option<String>,
    model_sha256: Option<String>,
    confidence: Option<f64>,
}

struct LocalModelRow {
    id: String,
    model_kind: String,
    file_path: String,
    file_size_bytes: String,
    sha256: String,
    version: String,
    input_format: String,
    model_card_id: String,
    license_id: String,
    license_confirmed_at: String,
    imported_at: String,
}

struct InferenceGapRecord {
    gap: InferenceGap,
    audit_event_id: Uuid,
}

struct InferenceGapRow {
    id: String,
    session_id: String,
    runtime_id: String,
    capture_segment_id: String,
    job_id: Option<String>,
    started_monotonic_ns: String,
    started_wall_clock: String,
    ended_monotonic_ns: String,
    ended_wall_clock: String,
    stage: String,
    reason: String,
    audit_event_id: String,
}

#[derive(Clone, Debug)]
struct StoredSpeakerCluster {
    record: SpeakerClusterRecord,
    audit_event_id: Uuid,
}

#[derive(Clone, Debug)]
struct StoredSpeakerClusterLabelRevision {
    revision: SpeakerClusterLabelRevision,
    audit_event_id: Uuid,
}

#[derive(Clone, Debug)]
struct StoredSpeakerClusterAliasRevision {
    revision: SpeakerClusterAliasRevision,
    audit_event_id: Uuid,
}

struct SpeakerClusterRow {
    id: String,
    session_id: String,
    ordinal: i64,
    audit_event_id: String,
}

struct SpeakerClusterLabelRevisionRow {
    id: String,
    speaker_cluster_id: String,
    parent_revision_id: Option<String>,
    revision: i64,
    label: String,
    is_user_named: i64,
    audit_event_id: String,
}

struct SpeakerClusterAliasRevisionRow {
    id: String,
    speaker_cluster_id: String,
    parent_revision_id: Option<String>,
    revision: i64,
    merged_into_cluster_id: Option<String>,
    audit_event_id: String,
}

struct SpeakerCatalog {
    clusters: BTreeMap<String, StoredSpeakerCluster>,
    labels: BTreeMap<String, Vec<StoredSpeakerClusterLabelRevision>>,
    aliases: BTreeMap<String, Vec<StoredSpeakerClusterAliasRevision>>,
}

impl SpeakerCatalog {
    fn latest_label(&self, cluster_id: &str) -> Option<&StoredSpeakerClusterLabelRevision> {
        self.labels
            .get(cluster_id)
            .and_then(|revisions| revisions.last())
    }

    fn latest_alias(&self, cluster_id: &str) -> Option<&StoredSpeakerClusterAliasRevision> {
        self.aliases
            .get(cluster_id)
            .and_then(|revisions| revisions.last())
    }

    fn active_aliases(&self) -> BTreeMap<String, Option<String>> {
        self.aliases
            .iter()
            .filter_map(|(cluster_id, revisions)| {
                revisions.last().map(|revision| {
                    (
                        cluster_id.clone(),
                        revision.revision.merged_into_cluster_id.clone(),
                    )
                })
            })
            .collect()
    }
}

fn local_model_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<LocalModelRow> {
    Ok(LocalModelRow {
        id: row.get(0)?,
        model_kind: row.get(1)?,
        file_path: row.get(2)?,
        file_size_bytes: row.get(3)?,
        sha256: row.get(4)?,
        version: row.get(5)?,
        input_format: row.get(6)?,
        model_card_id: row.get(7)?,
        license_id: row.get(8)?,
        license_confirmed_at: row.get(9)?,
        imported_at: row.get(10)?,
    })
}

fn inference_gap_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<InferenceGapRow> {
    Ok(InferenceGapRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        runtime_id: row.get(2)?,
        capture_segment_id: row.get(3)?,
        job_id: row.get(4)?,
        started_monotonic_ns: row.get(5)?,
        started_wall_clock: row.get(6)?,
        ended_monotonic_ns: row.get(7)?,
        ended_wall_clock: row.get(8)?,
        stage: row.get(9)?,
        reason: row.get(10)?,
        audit_event_id: row.get(11)?,
    })
}

fn speaker_cluster_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpeakerClusterRow> {
    Ok(SpeakerClusterRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        ordinal: row.get(2)?,
        audit_event_id: row.get(3)?,
    })
}

fn speaker_cluster_label_revision_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<SpeakerClusterLabelRevisionRow> {
    Ok(SpeakerClusterLabelRevisionRow {
        id: row.get(0)?,
        speaker_cluster_id: row.get(1)?,
        parent_revision_id: row.get(2)?,
        revision: row.get(3)?,
        label: row.get(4)?,
        is_user_named: row.get(5)?,
        audit_event_id: row.get(6)?,
    })
}

fn speaker_cluster_alias_revision_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<SpeakerClusterAliasRevisionRow> {
    Ok(SpeakerClusterAliasRevisionRow {
        id: row.get(0)?,
        speaker_cluster_id: row.get(1)?,
        parent_revision_id: row.get(2)?,
        revision: row.get(3)?,
        merged_into_cluster_id: row.get(4)?,
        audit_event_id: row.get(5)?,
    })
}

fn transcript_revision_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TranscriptRevisionRow> {
    Ok(TranscriptRevisionRow {
        id: row.get(0)?,
        logical_span_id: row.get(1)?,
        parent_revision_id: row.get(2)?,
        session_id: row.get(3)?,
        capture_start_ns: row.get(4)?,
        capture_end_ns: row.get(5)?,
        wall_clock_start: row.get(6)?,
        wall_clock_end: row.get(7)?,
        speaker_cluster_id: row.get(8)?,
        text: row.get(9)?,
        is_final: row.get(10)?,
        revision: row.get(11)?,
        source: row.get(12)?,
        model_provider: row.get(13)?,
        model_id: row.get(14)?,
        model_version: row.get(15)?,
        model_sha256: row.get(16)?,
        confidence: row.get(17)?,
    })
}

fn parse_transcript_revision(
    stored: TranscriptRevisionRow,
) -> Result<TranscriptRevision, AuditStoreError> {
    let is_final = match stored.is_final {
        0 => false,
        1 => true,
        value => {
            return Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "final flag",
                value: value.to_string(),
            });
        }
    };
    let model = parse_transcript_model_provenance(
        stored.model_provider,
        stored.model_id,
        stored.model_version,
        stored.model_sha256,
    )?;
    let source: TranscriptSource = serde_json::from_str(&stored.source).map_err(|_| {
        AuditStoreError::InvalidTranscriptMetadata {
            field: "source",
            value: stored.source,
        }
    })?;
    let revision = TranscriptRevision {
        id: parse_uuid(&stored.id)?,
        logical_span_id: parse_uuid(&stored.logical_span_id)?,
        parent_revision_id: parse_optional_uuid(stored.parent_revision_id)?,
        session_id: parse_uuid(&stored.session_id)?,
        capture_start_ns: parse_transcript_monotonic_ns(&stored.capture_start_ns)?,
        capture_end_ns: parse_transcript_monotonic_ns(&stored.capture_end_ns)?,
        wall_clock_start: parse_timestamp(&stored.wall_clock_start)?,
        wall_clock_end: parse_timestamp(&stored.wall_clock_end)?,
        speaker_cluster_id: stored.speaker_cluster_id,
        text: stored.text,
        is_final,
        revision: parse_transcript_revision_number(stored.revision)?,
        source,
        model,
        confidence: stored.confidence,
    };
    validate_transcript_revision_metadata(&revision)?;
    Ok(revision)
}

fn load_transcript_revision(
    connection: &Connection,
    revision_id: Uuid,
) -> Result<Option<TranscriptRevision>, AuditStoreError> {
    let stored = connection
        .query_row(
            "
            SELECT id, logical_span_id, parent_revision_id, session_id,
                   capture_start_ns, capture_end_ns, wall_clock_start, wall_clock_end,
                   speaker_cluster_id, text, is_final, revision, source,
                   model_provider, model_id, model_version, model_sha256, confidence
            FROM transcript_revisions
            WHERE id = ?1
            ",
            params![revision_id.to_string()],
            transcript_revision_row,
        )
        .optional()?;
    stored.map(parse_transcript_revision).transpose()
}

fn parse_registered_model(stored: LocalModelRow) -> Result<RegisteredModel, AuditStoreError> {
    let model = RegisteredModel {
        id: parse_uuid(&stored.id)?,
        model_kind: serde_json::from_str::<LocalModelKind>(&stored.model_kind)
            .map_err(|_| AuditStoreError::InvalidModelKind(stored.model_kind))?,
        file_path: stored.file_path.into(),
        file_size_bytes: stored.file_size_bytes.parse().map_err(|_| {
            AuditStoreError::InvalidModelMetadata {
                field: "file size",
                value: stored.file_size_bytes,
            }
        })?,
        sha256: stored.sha256,
        version: stored.version,
        input_format: stored.input_format,
        model_card_id: stored.model_card_id,
        license_id: stored.license_id,
        license_confirmed_at: parse_timestamp(&stored.license_confirmed_at)?,
        imported_at: parse_timestamp(&stored.imported_at)?,
    };
    validate_registered_model(&model)?;
    Ok(model)
}

fn parse_inference_gap_record(
    stored: InferenceGapRow,
) -> Result<InferenceGapRecord, AuditStoreError> {
    let stage = serde_json::from_str::<InferenceGapStage>(&stored.stage).map_err(|_| {
        AuditStoreError::InvalidInferenceGapMetadata {
            field: "stage",
            value: stored.stage,
        }
    })?;
    let reason = serde_json::from_str::<InferenceGapReason>(&stored.reason).map_err(|_| {
        AuditStoreError::InvalidInferenceGapMetadata {
            field: "reason",
            value: stored.reason,
        }
    })?;
    let gap = InferenceGap {
        id: parse_uuid(&stored.id)?,
        session_id: parse_uuid(&stored.session_id)?,
        runtime_id: parse_uuid(&stored.runtime_id)?,
        capture_segment_id: parse_uuid(&stored.capture_segment_id)?,
        job_id: parse_optional_uuid(stored.job_id)?,
        started_at: CapturePoint {
            monotonic_ns: parse_inference_gap_monotonic_ns(&stored.started_monotonic_ns)?,
            wall_clock: parse_timestamp(&stored.started_wall_clock)?,
        },
        ended_at: CapturePoint {
            monotonic_ns: parse_inference_gap_monotonic_ns(&stored.ended_monotonic_ns)?,
            wall_clock: parse_timestamp(&stored.ended_wall_clock)?,
        },
        stage,
        reason,
    };
    gap.validate()
        .map_err(|value| AuditStoreError::InvalidInferenceGapMetadata {
            field: "gap",
            value,
        })?;
    Ok(InferenceGapRecord {
        gap,
        audit_event_id: parse_uuid(&stored.audit_event_id)?,
    })
}

fn load_speaker_catalog(connection: &Connection) -> Result<SpeakerCatalog, AuditStoreError> {
    let mut clusters = BTreeMap::new();
    for stored in list_stored_speaker_clusters(connection)? {
        if clusters.insert(stored.record.id.clone(), stored).is_some() {
            return Err(speaker_error("speaker cluster", "duplicate cluster ID"));
        }
    }

    let mut labels = BTreeMap::<String, Vec<StoredSpeakerClusterLabelRevision>>::new();
    for stored in list_stored_speaker_cluster_label_revisions(connection)? {
        let cluster = clusters
            .get(&stored.revision.speaker_cluster_id)
            .ok_or_else(|| {
                speaker_error(
                    "speaker label cluster",
                    format!("missing {}", stored.revision.speaker_cluster_id),
                )
            })?;
        stored
            .revision
            .validate_for_cluster(&cluster.record)
            .map_err(|value| speaker_error("speaker label revision", value))?;
        labels
            .entry(stored.revision.speaker_cluster_id.clone())
            .or_default()
            .push(stored);
    }

    for cluster in clusters.values() {
        let revisions = labels.get_mut(&cluster.record.id).ok_or_else(|| {
            speaker_error("speaker label revision", "cluster has no initial label")
        })?;
        revisions.sort_by_key(|stored| stored.revision.revision);
        let initial = revisions.first().ok_or_else(|| {
            speaker_error("speaker label revision", "cluster has no initial label")
        })?;
        SpeakerClusterCreatedAuditPayload::new(cluster.record.clone(), initial.revision.clone())
            .map_err(|value| speaker_error("initial speaker label", value))?;
        if initial.audit_event_id != cluster.audit_event_id {
            return Err(speaker_error(
                "initial speaker label audit binding",
                "does not match cluster creation event",
            ));
        }
        for revisions in revisions.windows(2) {
            revisions[1]
                .revision
                .validate_successor_of(&revisions[0].revision)
                .map_err(|value| speaker_error("speaker label revision chain", value))?;
        }
    }

    let mut aliases = BTreeMap::<String, Vec<StoredSpeakerClusterAliasRevision>>::new();
    for stored in list_stored_speaker_cluster_alias_revisions(connection)? {
        let cluster = clusters
            .get(&stored.revision.speaker_cluster_id)
            .ok_or_else(|| {
                speaker_error(
                    "speaker alias cluster",
                    format!("missing {}", stored.revision.speaker_cluster_id),
                )
            })?;
        stored
            .revision
            .validate_for_cluster(&cluster.record)
            .map_err(|value| speaker_error("speaker alias revision", value))?;
        if let Some(target_cluster_id) = &stored.revision.merged_into_cluster_id {
            let target = clusters.get(target_cluster_id).ok_or_else(|| {
                speaker_error(
                    "speaker alias target",
                    format!("missing {target_cluster_id}"),
                )
            })?;
            if target.record.session_id != cluster.record.session_id {
                return Err(speaker_error(
                    "speaker alias session",
                    format!("{} -> {target_cluster_id}", cluster.record.id),
                ));
            }
        }
        aliases
            .entry(stored.revision.speaker_cluster_id.clone())
            .or_default()
            .push(stored);
    }
    for revisions in aliases.values_mut() {
        revisions.sort_by_key(|stored| stored.revision.revision);
        let initial = revisions.first().ok_or_else(|| {
            speaker_error("speaker alias revision", "missing initial alias revision")
        })?;
        if initial.revision.revision != 1 || initial.revision.parent_revision_id.is_some() {
            return Err(speaker_error(
                "speaker alias revision chain",
                "first alias revision must be revision one without a parent",
            ));
        }
        for revisions in revisions.windows(2) {
            revisions[1]
                .revision
                .validate_successor_of(&revisions[0].revision)
                .map_err(|value| speaker_error("speaker alias revision chain", value))?;
        }
    }

    let catalog = SpeakerCatalog {
        clusters,
        labels,
        aliases,
    };
    validate_active_speaker_aliases(&catalog.clusters, &catalog.active_aliases())?;
    Ok(catalog)
}

fn list_stored_speaker_clusters(
    connection: &Connection,
) -> Result<Vec<StoredSpeakerCluster>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT id, session_id, ordinal, audit_event_id
        FROM speaker_clusters
        ORDER BY sequence ASC
        ",
    )?;
    let rows = statement.query_map([], speaker_cluster_row)?;
    rows.map(|row| parse_stored_speaker_cluster(row?)).collect()
}

fn list_stored_speaker_cluster_label_revisions(
    connection: &Connection,
) -> Result<Vec<StoredSpeakerClusterLabelRevision>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT id, speaker_cluster_id, parent_revision_id, revision, label, is_user_named,
               audit_event_id
        FROM speaker_cluster_label_revisions
        ORDER BY sequence ASC
        ",
    )?;
    let rows = statement.query_map([], speaker_cluster_label_revision_row)?;
    rows.map(|row| parse_stored_speaker_cluster_label_revision(row?))
        .collect()
}

fn list_stored_speaker_cluster_alias_revisions(
    connection: &Connection,
) -> Result<Vec<StoredSpeakerClusterAliasRevision>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT id, speaker_cluster_id, parent_revision_id, revision, merged_into_cluster_id,
               audit_event_id
        FROM speaker_cluster_alias_revisions
        ORDER BY sequence ASC
        ",
    )?;
    let rows = statement.query_map([], speaker_cluster_alias_revision_row)?;
    rows.map(|row| parse_stored_speaker_cluster_alias_revision(row?))
        .collect()
}

fn parse_stored_speaker_cluster(
    stored: SpeakerClusterRow,
) -> Result<StoredSpeakerCluster, AuditStoreError> {
    let record = SpeakerClusterRecord {
        id: stored.id,
        session_id: parse_uuid(&stored.session_id)?,
        ordinal: parse_speaker_u32("speaker cluster ordinal", stored.ordinal)?,
    };
    record
        .validate()
        .map_err(|value| speaker_error("speaker cluster", value))?;
    Ok(StoredSpeakerCluster {
        record,
        audit_event_id: parse_uuid(&stored.audit_event_id)?,
    })
}

fn parse_stored_speaker_cluster_label_revision(
    stored: SpeakerClusterLabelRevisionRow,
) -> Result<StoredSpeakerClusterLabelRevision, AuditStoreError> {
    let is_user_named = parse_speaker_bool("speaker label user-named flag", stored.is_user_named)?;
    let revision = SpeakerClusterLabelRevision {
        id: parse_uuid(&stored.id)?,
        speaker_cluster_id: stored.speaker_cluster_id,
        parent_revision_id: parse_optional_uuid(stored.parent_revision_id)?,
        revision: parse_speaker_u32("speaker label revision", stored.revision)?,
        label: stored.label,
        is_user_named,
    };
    revision
        .validate()
        .map_err(|value| speaker_error("speaker label revision", value))?;
    Ok(StoredSpeakerClusterLabelRevision {
        revision,
        audit_event_id: parse_uuid(&stored.audit_event_id)?,
    })
}

fn parse_stored_speaker_cluster_alias_revision(
    stored: SpeakerClusterAliasRevisionRow,
) -> Result<StoredSpeakerClusterAliasRevision, AuditStoreError> {
    let revision = SpeakerClusterAliasRevision {
        id: parse_uuid(&stored.id)?,
        speaker_cluster_id: stored.speaker_cluster_id,
        parent_revision_id: parse_optional_uuid(stored.parent_revision_id)?,
        revision: parse_speaker_u32("speaker alias revision", stored.revision)?,
        merged_into_cluster_id: stored.merged_into_cluster_id,
    };
    revision
        .validate()
        .map_err(|value| speaker_error("speaker alias revision", value))?;
    Ok(StoredSpeakerClusterAliasRevision {
        revision,
        audit_event_id: parse_uuid(&stored.audit_event_id)?,
    })
}

fn parse_transcript_model_provenance(
    provider: Option<String>,
    model_id: Option<String>,
    version: Option<String>,
    sha256: Option<String>,
) -> Result<Option<TranscriptModelProvenance>, AuditStoreError> {
    match (provider, model_id, version, sha256) {
        (None, None, None, None) => Ok(None),
        (Some(provider), Some(model_id), Some(version), sha256) => {
            let model = TranscriptModelProvenance {
                provider,
                model_id,
                version,
                sha256,
            };
            model
                .validate()
                .map_err(|value| AuditStoreError::InvalidTranscriptMetadata {
                    field: "model provenance",
                    value,
                })?;
            Ok(Some(model))
        }
        values => Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "model provenance",
            value: format!(
                "provider={:?}, model_id={:?}, version={:?}, sha256={:?}",
                values.0, values.1, values.2, values.3
            ),
        }),
    }
}

fn validate_transcript_revision_metadata(
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    if revision.capture_end_ns < revision.capture_start_ns {
        return Err(AuditStoreError::InvalidTranscriptRange);
    }
    if revision.wall_clock_end < revision.wall_clock_start {
        return Err(AuditStoreError::InvalidTranscriptWallClockRange);
    }
    if !revision.is_final {
        return Err(AuditStoreError::NonFinalTranscript);
    }
    revision
        .validate()
        .map_err(|value| AuditStoreError::InvalidTranscriptMetadata {
            field: "revision",
            value,
        })
}

fn validate_transcript_revision(
    connection: &Connection,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_transcript_revision_metadata(revision)?;

    let Some(parent_revision_id) = revision.parent_revision_id else {
        return Ok(());
    };
    let parent = connection
        .query_row(
            "
            SELECT logical_span_id, session_id, revision
            FROM transcript_revisions
            WHERE id = ?1
            ",
            params![parent_revision_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((parent_logical_span_id, parent_session_id, parent_revision)) = parent else {
        return Err(AuditStoreError::MissingTranscriptParent(
            parent_revision_id.to_string(),
        ));
    };
    let parent_logical_span_id = parse_uuid(&parent_logical_span_id)?;
    let parent_session_id = parse_uuid(&parent_session_id)?;
    if parent_logical_span_id != revision.logical_span_id {
        return Err(AuditStoreError::InvalidTranscriptParent {
            parent_id: parent_revision_id.to_string(),
            reason: "logical span ID differs",
        });
    }
    if parent_session_id != revision.session_id {
        return Err(AuditStoreError::InvalidTranscriptParent {
            parent_id: parent_revision_id.to_string(),
            reason: "session ID differs",
        });
    }
    let parent_revision = parse_transcript_revision_number(parent_revision)?;
    if parent_revision.checked_add(1) != Some(revision.revision) {
        return Err(AuditStoreError::InvalidTranscriptParent {
            parent_id: parent_revision_id.to_string(),
            reason: "revision number must increase by exactly one",
        });
    }
    Ok(())
}

fn validate_transcript_audit_event(
    event: &AuditEvent,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    if revision.source == TranscriptSource::LocalInference {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "local inference audit binding",
            value: "native ASR transcript revisions must use the idempotency-bound write path"
                .to_owned(),
        });
    }
    validate_transcript_audit_event_metadata(event, revision)?;
    if !event.matches_payload(revision).map_err(|error| {
        AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event payload",
            value: error.to_string(),
        }
    })? {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event payload",
            value: "digest does not match transcript revision".to_owned(),
        });
    }
    Ok(())
}

fn reject_speaker_reassignment_from_generic_path(
    connection: &Connection,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    let Some(parent_revision_id) = revision.parent_revision_id else {
        return Ok(());
    };
    let Some(parent) = load_transcript_revision(connection, parent_revision_id)? else {
        return Ok(());
    };
    if is_speaker_only_reassignment(revision, &parent) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit binding",
            value: "speaker-only corrections must use the dedicated audit write path".to_owned(),
        });
    }
    Ok(())
}

fn validate_transcript_speaker_reassignment_with_audit(
    connection: &Connection,
    event: &AuditEvent,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_transcript_revision(connection, revision)?;
    let parent_revision_id =
        revision
            .parent_revision_id
            .ok_or_else(|| AuditStoreError::InvalidTranscriptMetadata {
                field: "speaker reassignment parent",
                value: "speaker reassignment must append to a prior revision".to_owned(),
            })?;
    let parent = load_transcript_revision(connection, parent_revision_id)?
        .ok_or_else(|| AuditStoreError::MissingTranscriptParent(parent_revision_id.to_string()))?;
    validate_transcript_speaker_reassignment_audit_event(connection, event, revision, &parent)
}

fn validate_transcript_speaker_reassignment_audit_event(
    connection: &Connection,
    event: &AuditEvent,
    revision: &TranscriptRevision,
    parent: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_transcript_speaker_reassignment_revision(revision, parent)?;
    validate_transcript_speaker_reassignment_target(connection, revision)?;
    if event.kind != AuditKind::TranscriptSpeakerReassigned {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit event kind",
            value: serde_json::to_string(&event.kind).expect("audit kind serializes"),
        });
    }
    if event.run_id != Some(revision.session_id) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit event session linkage",
            value: format!("run_id={:?}", event.run_id),
        });
    }
    if event.causation_id != revision.parent_revision_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit event causation linkage",
            value: format!("causation_id={:?}", event.causation_id),
        });
    }
    if !event.matches_payload(revision).map_err(|error| {
        AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit event payload",
            value: error.to_string(),
        }
    })? {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment audit event payload",
            value: "digest does not match transcript revision".to_owned(),
        });
    }
    Ok(())
}

/// A dedicated speaker reassignment may reference only an active catalog
/// entry in the same session. Generic transcript revisions intentionally do
/// not use this check so pre-M3 strings such as `speaker-1` remain readable.
fn validate_transcript_speaker_reassignment_target(
    connection: &Connection,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    let Some(target_cluster_id) = revision.speaker_cluster_id.as_deref() else {
        return Ok(());
    };
    let catalog = load_speaker_catalog(connection)?;
    let target = catalog.clusters.get(target_cluster_id).ok_or_else(|| {
        AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment target",
            value: "speaker cluster does not exist in the transcript session".to_owned(),
        }
    })?;
    if target.record.session_id != revision.session_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment target",
            value: "speaker cluster does not belong to the transcript session".to_owned(),
        });
    }

    let canonical_cluster_id = resolve_speaker_cluster_canonical_id(
        target_cluster_id,
        &catalog.clusters,
        &catalog.active_aliases(),
    )?;
    if canonical_cluster_id != target_cluster_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment target",
            value: "speaker cluster is merged and cannot receive assignments".to_owned(),
        });
    }
    Ok(())
}

fn validate_transcript_speaker_reassignment_revision(
    revision: &TranscriptRevision,
    parent: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_transcript_revision_metadata(revision)?;
    if revision.source != TranscriptSource::UserEdited {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment source",
            value: format!("{:?}", revision.source),
        });
    }
    if revision.parent_revision_id != Some(parent.id) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment parent",
            value: format!("parent_revision_id={:?}", revision.parent_revision_id),
        });
    }
    if !speaker_reassignment_chain_matches(revision, parent) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment revision chain",
            value: "logical span, session, and revision number must follow the parent".to_owned(),
        });
    }
    if !speaker_reassignment_facts_match(revision, parent) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment immutable transcript facts",
            value: "text, timing, finality, model, and confidence must match the parent".to_owned(),
        });
    }
    if revision.speaker_cluster_id == parent.speaker_cluster_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "speaker reassignment target",
            value: "speaker cluster assignment must change".to_owned(),
        });
    }
    Ok(())
}

fn is_speaker_only_reassignment(
    revision: &TranscriptRevision,
    parent: &TranscriptRevision,
) -> bool {
    revision.source == TranscriptSource::UserEdited
        && revision.parent_revision_id == Some(parent.id)
        && speaker_reassignment_chain_matches(revision, parent)
        && speaker_reassignment_facts_match(revision, parent)
        && revision.speaker_cluster_id != parent.speaker_cluster_id
}

fn speaker_reassignment_chain_matches(
    revision: &TranscriptRevision,
    parent: &TranscriptRevision,
) -> bool {
    revision.logical_span_id == parent.logical_span_id
        && revision.session_id == parent.session_id
        && parent.revision.checked_add(1) == Some(revision.revision)
}

fn speaker_reassignment_facts_match(
    revision: &TranscriptRevision,
    parent: &TranscriptRevision,
) -> bool {
    revision.capture_start_ns == parent.capture_start_ns
        && revision.capture_end_ns == parent.capture_end_ns
        && revision.wall_clock_start == parent.wall_clock_start
        && revision.wall_clock_end == parent.wall_clock_end
        && revision.text == parent.text
        && revision.is_final == parent.is_final
        && revision.model == parent.model
        && revision.confidence == parent.confidence
}

fn validate_asr_final_audit_event(
    event: &AuditEvent,
    revision: &TranscriptRevision,
    idempotency: &AsrFinalIdempotencyBinding,
) -> Result<(), AuditStoreError> {
    validate_asr_final_idempotency_binding(idempotency, revision)?;
    validate_transcript_audit_event_metadata(event, revision)?;
    let payload = AsrFinalAuditPayload::new(revision, idempotency);
    if !event.matches_payload(&payload).map_err(|error| {
        AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event payload",
            value: error.to_string(),
        }
    })? {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event payload",
            value: "digest does not match transcript revision and ASR idempotency binding"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_transcript_audit_event_metadata(
    event: &AuditEvent,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    if event.kind != AuditKind::TranscriptRevisionRecorded {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event kind",
            value: serde_json::to_string(&event.kind).expect("audit kind serializes"),
        });
    }
    if event.run_id != Some(revision.session_id) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event session linkage",
            value: format!("run_id={:?}", event.run_id),
        });
    }
    if event.causation_id != revision.parent_revision_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event causation linkage",
            value: format!("causation_id={:?}", event.causation_id),
        });
    }
    if event.monotonic_ns != revision.capture_end_ns || event.wall_clock != revision.wall_clock_end
    {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "audit event capture endpoint",
            value: format!(
                "monotonic_ns={}, wall_clock={}",
                event.monotonic_ns,
                event.wall_clock.to_rfc3339()
            ),
        });
    }
    Ok(())
}

fn validate_asr_final_idempotency_binding(
    idempotency: &AsrFinalIdempotencyBinding,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_persisted_asr_final_idempotency_binding(idempotency)?;
    if idempotency.session_id != revision.session_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency session",
            value: format!(
                "binding session {}, revision session {}",
                idempotency.session_id, revision.session_id
            ),
        });
    }
    if idempotency.revision_id != revision.id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency revision ID",
            value: format!(
                "binding revision {}, transcript revision {}",
                idempotency.revision_id, revision.id
            ),
        });
    }
    if idempotency.logical_span_id != revision.logical_span_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency logical span",
            value: format!(
                "binding logical span {}, transcript logical span {}",
                idempotency.logical_span_id, revision.logical_span_id
            ),
        });
    }
    if revision.id != revision.logical_span_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency revision",
            value: "native ASR final must create the first durable revision".to_owned(),
        });
    }
    if revision.source != TranscriptSource::LocalInference {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency source",
            value: format!("{:?}", revision.source),
        });
    }
    Ok(())
}

fn validate_persisted_asr_final_idempotency_binding(
    idempotency: &AsrFinalIdempotencyBinding,
) -> Result<(), AuditStoreError> {
    if idempotency.emission_revision == 0 {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency revision",
            value: "0".to_owned(),
        });
    }
    validate_sha256_hex(
        "ASR final utterance key SHA-256",
        &idempotency.utterance_key_sha256,
    )?;
    validate_sha256_hex(
        "ASR final payload SHA-256",
        &idempotency.emission_payload_sha256,
    )?;
    let expected_logical_span_id = logical_span_id_for_asr_utterance_digest(
        idempotency.session_id,
        &idempotency.utterance_key_sha256,
    );
    if idempotency.logical_span_id != expected_logical_span_id {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency logical span",
            value: format!(
                "expected {expected_logical_span_id}, received {}",
                idempotency.logical_span_id
            ),
        });
    }
    Ok(())
}

fn validate_local_model_audit_event(
    event: &AuditEvent,
    model: &RegisteredModel,
) -> Result<(), AuditStoreError> {
    if event.kind != AuditKind::LocalModelImported {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "audit event kind",
            value: serde_json::to_string(&event.kind).expect("audit kind serializes"),
        });
    }
    if !event
        .matches_payload(model)
        .map_err(|error| AuditStoreError::InvalidModelMetadata {
            field: "audit event payload",
            value: error.to_string(),
        })?
    {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "audit event payload",
            value: "digest does not match local model registration".to_owned(),
        });
    }
    if event.run_id.is_some() {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "audit event session linkage",
            value: format!("run_id={:?}", event.run_id),
        });
    }
    if event.causation_id.is_some() {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "audit event causation linkage",
            value: format!("causation_id={:?}", event.causation_id),
        });
    }
    Ok(())
}

fn validate_inference_gap(gap: &InferenceGap) -> Result<(), AuditStoreError> {
    gap.validate()
        .map_err(|value| AuditStoreError::InvalidInferenceGapMetadata {
            field: "gap",
            value,
        })
}

fn validate_inference_gap_audit_event(
    event: &AuditEvent,
    gap: &InferenceGap,
) -> Result<(), AuditStoreError> {
    validate_inference_gap(gap)?;
    if event.kind != AuditKind::InferenceGapRecorded {
        return Err(AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event kind",
            value: serde_json::to_string(&event.kind).expect("audit kind serializes"),
        });
    }
    if event.run_id != Some(gap.session_id) {
        return Err(AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event session linkage",
            value: format!("run_id={:?}", event.run_id),
        });
    }
    if event.causation_id != gap.job_id {
        return Err(AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event causation linkage",
            value: format!("causation_id={:?}", event.causation_id),
        });
    }
    if event.monotonic_ns != gap.ended_at.monotonic_ns
        || event.wall_clock != gap.ended_at.wall_clock
    {
        return Err(AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event capture endpoint",
            value: format!(
                "monotonic_ns={}, wall_clock={}",
                event.monotonic_ns,
                event.wall_clock.to_rfc3339()
            ),
        });
    }
    if !event.matches_payload(gap).map_err(|error| {
        AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event payload",
            value: error.to_string(),
        }
    })? {
        return Err(AuditStoreError::InvalidInferenceGapMetadata {
            field: "audit event payload",
            value: "digest does not match inference gap".to_owned(),
        });
    }
    Ok(())
}

fn validate_speaker_cluster_created_audit_event(
    event: &AuditEvent,
    cluster: &SpeakerClusterRecord,
    initial_label: &SpeakerClusterLabelRevision,
) -> Result<(), AuditStoreError> {
    let payload = SpeakerClusterCreatedAuditPayload::new(cluster.clone(), initial_label.clone())
        .map_err(|value| speaker_error("speaker cluster creation", value))?;
    validate_speaker_audit_event_metadata(
        event,
        AuditKind::SpeakerClusterCreated,
        cluster.session_id,
        None,
        &payload,
    )
}

fn validate_speaker_cluster_label_revision_for_write(
    connection: &Connection,
    revision: &SpeakerClusterLabelRevision,
) -> Result<SpeakerClusterRecord, AuditStoreError> {
    let catalog = load_speaker_catalog(connection)?;
    let cluster = catalog
        .clusters
        .get(&revision.speaker_cluster_id)
        .ok_or_else(|| {
            speaker_error(
                "speaker label cluster",
                format!("missing {}", revision.speaker_cluster_id),
            )
        })?;
    revision
        .validate_for_cluster(&cluster.record)
        .map_err(|value| speaker_error("speaker label revision", value))?;
    let previous = catalog
        .latest_label(&revision.speaker_cluster_id)
        .ok_or_else(|| speaker_error("speaker label parent", "cluster has no initial label"))?;
    revision
        .validate_successor_of(&previous.revision)
        .map_err(|value| speaker_error("speaker label parent", value))?;
    Ok(cluster.record.clone())
}

fn validate_speaker_cluster_label_revision_audit_event(
    event: &AuditEvent,
    cluster: &SpeakerClusterRecord,
    revision: &SpeakerClusterLabelRevision,
) -> Result<(), AuditStoreError> {
    validate_speaker_audit_event_metadata(
        event,
        AuditKind::SpeakerClusterLabelRevisionRecorded,
        cluster.session_id,
        revision.parent_revision_id,
        revision,
    )
}

fn validate_speaker_cluster_alias_revision_for_write(
    connection: &Connection,
    revision: &SpeakerClusterAliasRevision,
) -> Result<SpeakerClusterRecord, AuditStoreError> {
    let catalog = load_speaker_catalog(connection)?;
    let cluster = catalog
        .clusters
        .get(&revision.speaker_cluster_id)
        .ok_or_else(|| {
            speaker_error(
                "speaker alias cluster",
                format!("missing {}", revision.speaker_cluster_id),
            )
        })?;
    revision
        .validate_for_cluster(&cluster.record)
        .map_err(|value| speaker_error("speaker alias revision", value))?;

    match catalog.latest_alias(&revision.speaker_cluster_id) {
        Some(previous) => revision
            .validate_successor_of(&previous.revision)
            .map_err(|value| speaker_error("speaker alias parent", value))?,
        None if revision.revision != 1 || revision.parent_revision_id.is_some() => {
            return Err(speaker_error(
                "speaker alias parent",
                "first alias revision must be revision one without a parent",
            ));
        }
        None => {}
    }

    let mut aliases = catalog.active_aliases();
    aliases.insert(
        revision.speaker_cluster_id.clone(),
        revision.merged_into_cluster_id.clone(),
    );
    validate_active_speaker_aliases(&catalog.clusters, &aliases)?;
    Ok(cluster.record.clone())
}

fn validate_speaker_cluster_alias_revision_audit_event(
    event: &AuditEvent,
    cluster: &SpeakerClusterRecord,
    revision: &SpeakerClusterAliasRevision,
) -> Result<(), AuditStoreError> {
    validate_speaker_audit_event_metadata(
        event,
        AuditKind::SpeakerClusterAliasRevisionRecorded,
        cluster.session_id,
        revision.parent_revision_id,
        revision,
    )
}

fn validate_speaker_audit_event_metadata<T: Serialize>(
    event: &AuditEvent,
    kind: AuditKind,
    session_id: Uuid,
    causation_id: Option<Uuid>,
    payload: &T,
) -> Result<(), AuditStoreError> {
    if event.kind != kind {
        return Err(speaker_error(
            "audit event kind",
            serde_json::to_string(&event.kind).expect("audit kind serializes"),
        ));
    }
    if event.run_id != Some(session_id) {
        return Err(speaker_error(
            "audit event session linkage",
            format!("run_id={:?}", event.run_id),
        ));
    }
    if event.causation_id != causation_id {
        return Err(speaker_error(
            "audit event causation linkage",
            format!("causation_id={:?}", event.causation_id),
        ));
    }
    if !event
        .matches_payload(payload)
        .map_err(|error| speaker_error("audit event payload", error.to_string()))?
    {
        return Err(speaker_error(
            "audit event payload",
            "digest does not match speaker catalog record",
        ));
    }
    Ok(())
}

fn validate_active_speaker_aliases(
    clusters: &BTreeMap<String, StoredSpeakerCluster>,
    aliases: &BTreeMap<String, Option<String>>,
) -> Result<(), AuditStoreError> {
    for (source_cluster_id, target_cluster_id) in aliases {
        let source = clusters.get(source_cluster_id).ok_or_else(|| {
            speaker_error(
                "speaker alias source",
                format!("missing {source_cluster_id}"),
            )
        })?;
        if let Some(target_cluster_id) = target_cluster_id {
            let target = clusters.get(target_cluster_id).ok_or_else(|| {
                speaker_error(
                    "speaker alias target",
                    format!("missing {target_cluster_id}"),
                )
            })?;
            if target.record.session_id != source.record.session_id {
                return Err(speaker_error(
                    "speaker alias session",
                    format!("{source_cluster_id} -> {target_cluster_id}"),
                ));
            }
        }
    }
    for cluster_id in aliases.keys() {
        resolve_speaker_cluster_canonical_id(cluster_id, clusters, aliases)?;
    }
    Ok(())
}

fn resolve_speaker_cluster_canonical_id(
    cluster_id: &str,
    clusters: &BTreeMap<String, StoredSpeakerCluster>,
    aliases: &BTreeMap<String, Option<String>>,
) -> Result<String, AuditStoreError> {
    if !clusters.contains_key(cluster_id) {
        return Err(speaker_error(
            "speaker cluster",
            format!("missing {cluster_id}"),
        ));
    }

    let mut current = cluster_id.to_owned();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current.clone()) {
            return Err(speaker_error("speaker alias cycle", current));
        }
        match aliases.get(&current).and_then(|target| target.as_ref()) {
            Some(target) => {
                if !clusters.contains_key(target) {
                    return Err(speaker_error(
                        "speaker alias target",
                        format!("missing {target}"),
                    ));
                }
                current = target.clone();
            }
            None => return Ok(current),
        }
    }
}

fn current_speaker_cluster_span_count(
    connection: &Connection,
    session_id: Uuid,
    cluster_id: &str,
) -> Result<i64, AuditStoreError> {
    connection
        .query_row(
            "
            SELECT COUNT(*)
            FROM transcript_revisions AS revisions
            WHERE revisions.session_id = ?1
              AND revisions.speaker_cluster_id = ?2
              AND revisions.is_final = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM transcript_revisions AS children
                  WHERE children.parent_revision_id = revisions.id
              )
            ",
            params![session_id.to_string(), cluster_id],
            |row| row.get(0),
        )
        .map_err(AuditStoreError::from)
}

fn verify_speaker_catalog(events: &[AuditEvent], catalog: &SpeakerCatalog) -> bool {
    let events_by_id = events
        .iter()
        .map(|event| (event.id, event))
        .collect::<BTreeMap<_, _>>();
    let creation_events = events
        .iter()
        .filter(|event| event.kind == AuditKind::SpeakerClusterCreated)
        .count();
    let label_events = events
        .iter()
        .filter(|event| event.kind == AuditKind::SpeakerClusterLabelRevisionRecorded)
        .count();
    let alias_events = events
        .iter()
        .filter(|event| event.kind == AuditKind::SpeakerClusterAliasRevisionRecorded)
        .count();
    let expected_label_events = catalog
        .labels
        .values()
        .map(|revisions| revisions.len().saturating_sub(1))
        .sum::<usize>();
    let expected_alias_events = catalog.aliases.values().map(Vec::len).sum::<usize>();
    if creation_events != catalog.clusters.len()
        || label_events != expected_label_events
        || alias_events != expected_alias_events
    {
        return false;
    }

    let mut bound_events = BTreeSet::new();
    for cluster in catalog.clusters.values() {
        let Some(initial_label) = catalog
            .labels
            .get(&cluster.record.id)
            .and_then(|revisions| revisions.first())
        else {
            return false;
        };
        let Some(event) = events_by_id.get(&cluster.audit_event_id) else {
            return false;
        };
        if !bound_events.insert(cluster.audit_event_id)
            || validate_speaker_cluster_created_audit_event(
                event,
                &cluster.record,
                &initial_label.revision,
            )
            .is_err()
        {
            return false;
        }
        for revision in catalog
            .labels
            .get(&cluster.record.id)
            .into_iter()
            .flatten()
            .skip(1)
        {
            let Some(event) = events_by_id.get(&revision.audit_event_id) else {
                return false;
            };
            if !bound_events.insert(revision.audit_event_id)
                || validate_speaker_cluster_label_revision_audit_event(
                    event,
                    &cluster.record,
                    &revision.revision,
                )
                .is_err()
            {
                return false;
            }
        }
    }
    for revisions in catalog.aliases.values() {
        for revision in revisions {
            let Some(cluster) = catalog.clusters.get(&revision.revision.speaker_cluster_id) else {
                return false;
            };
            let Some(event) = events_by_id.get(&revision.audit_event_id) else {
                return false;
            };
            if !bound_events.insert(revision.audit_event_id)
                || validate_speaker_cluster_alias_revision_audit_event(
                    event,
                    &cluster.record,
                    &revision.revision,
                )
                .is_err()
            {
                return false;
            }
        }
    }
    true
}

fn insert_audit_event(connection: &Connection, event: &AuditEvent) -> Result<(), AuditStoreError> {
    if !event.verifies() {
        return Err(AuditStoreError::Integrity);
    }
    let previous_hash = connection
        .query_row(
            "SELECT hash FROM audit_events ORDER BY sequence DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if event.previous_hash.as_deref() != previous_hash.as_deref() {
        return Err(AuditStoreError::Integrity);
    }
    connection.execute(
        "
        INSERT INTO audit_events (
            id, run_id, causation_id, kind, monotonic_ns, wall_clock, payload_hash, previous_hash, hash
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ",
        params![
            event.id.to_string(),
            event.run_id.map(|value| value.to_string()),
            event.causation_id.map(|value| value.to_string()),
            serde_json::to_string(&event.kind).expect("audit kind serializes"),
            event.monotonic_ns.to_string(),
            event.wall_clock.to_rfc3339(),
            event.payload_hash,
            event.previous_hash,
            event.hash,
        ],
    )?;
    Ok(())
}

fn insert_speaker_cluster(
    connection: &Connection,
    cluster: &SpeakerClusterRecord,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    cluster
        .validate()
        .map_err(|value| speaker_error("speaker cluster", value))?;
    connection.execute(
        "
        INSERT INTO speaker_clusters (id, session_id, ordinal, audit_event_id)
        VALUES (?1, ?2, ?3, ?4)
        ",
        params![
            &cluster.id,
            cluster.session_id.to_string(),
            i64::from(cluster.ordinal),
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_speaker_cluster_label_revision(
    connection: &Connection,
    revision: &SpeakerClusterLabelRevision,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    revision
        .validate()
        .map_err(|value| speaker_error("speaker label revision", value))?;
    connection.execute(
        "
        INSERT INTO speaker_cluster_label_revisions (
            id, speaker_cluster_id, parent_revision_id, revision, label, is_user_named,
            audit_event_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ",
        params![
            revision.id.to_string(),
            &revision.speaker_cluster_id,
            revision.parent_revision_id.map(|value| value.to_string()),
            i64::from(revision.revision),
            &revision.label,
            if revision.is_user_named { 1_i64 } else { 0_i64 },
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_speaker_cluster_alias_revision(
    connection: &Connection,
    revision: &SpeakerClusterAliasRevision,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    revision
        .validate()
        .map_err(|value| speaker_error("speaker alias revision", value))?;
    connection.execute(
        "
        INSERT INTO speaker_cluster_alias_revisions (
            id, speaker_cluster_id, parent_revision_id, revision, merged_into_cluster_id,
            audit_event_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ",
        params![
            revision.id.to_string(),
            &revision.speaker_cluster_id,
            revision.parent_revision_id.map(|value| value.to_string()),
            i64::from(revision.revision),
            revision.merged_into_cluster_id.as_deref(),
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_transcript_revision(
    connection: &Connection,
    revision: &TranscriptRevision,
) -> Result<(), AuditStoreError> {
    validate_transcript_revision(connection, revision)?;
    let model = revision.model.as_ref();
    connection.execute(
        "
        INSERT INTO transcript_revisions (
            id, logical_span_id, parent_revision_id, session_id,
            capture_start_ns, capture_end_ns, wall_clock_start, wall_clock_end,
            speaker_cluster_id, text, is_final, revision, source,
            model_provider, model_id, model_version, model_sha256, confidence
        ) VALUES (
            ?1, ?2, ?3, ?4,
            ?5, ?6, ?7, ?8,
            ?9, ?10, ?11, ?12, ?13,
            ?14, ?15, ?16, ?17, ?18
        )
        ",
        params![
            revision.id.to_string(),
            revision.logical_span_id.to_string(),
            revision.parent_revision_id.map(|value| value.to_string()),
            revision.session_id.to_string(),
            revision.capture_start_ns.to_string(),
            revision.capture_end_ns.to_string(),
            revision.wall_clock_start.to_rfc3339(),
            revision.wall_clock_end.to_rfc3339(),
            revision.speaker_cluster_id.as_deref(),
            &revision.text,
            if revision.is_final { 1_i64 } else { 0_i64 },
            i64::from(revision.revision),
            serde_json::to_string(&revision.source).expect("transcript source serializes"),
            model.map(|value| value.provider.as_str()),
            model.map(|value| value.model_id.as_str()),
            model.map(|value| value.version.as_str()),
            model.and_then(|value| value.sha256.as_deref()),
            revision.confidence,
        ],
    )?;
    connection.execute(
        "
        INSERT INTO transcript_revision_fts (
            text, revision_id, session_id, logical_span_id
        ) VALUES (?1, ?2, ?3, ?4)
        ",
        params![
            &revision.text,
            revision.id.to_string(),
            revision.session_id.to_string(),
            revision.logical_span_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_asr_final_idempotency(
    connection: &Connection,
    idempotency: &AsrFinalIdempotencyBinding,
    revision_id: Uuid,
) -> Result<(), AuditStoreError> {
    connection.execute(
        "
        INSERT INTO asr_final_idempotency (
            session_id, utterance_key_sha256, emission_revision, revision_id,
            emission_payload_sha256
        ) VALUES (?1, ?2, ?3, ?4, ?5)
        ",
        params![
            idempotency.session_id.to_string(),
            &idempotency.utterance_key_sha256,
            i64::from(idempotency.emission_revision),
            revision_id.to_string(),
            &idempotency.emission_payload_sha256,
        ],
    )?;
    Ok(())
}

fn validate_asr_final_idempotency_key(key: &AsrFinalIdempotencyKey) -> Result<(), AuditStoreError> {
    key.validate()
        .map_err(|value| AuditStoreError::InvalidTranscriptMetadata {
            field: "ASR final idempotency key",
            value,
        })
}

fn validate_sha256_hex(field: &'static str, value: &str) -> Result<(), AuditStoreError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AuditStoreError::InvalidTranscriptMetadata {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn insert_registered_model(
    connection: &Connection,
    model: &RegisteredModel,
) -> Result<(), AuditStoreError> {
    validate_registered_model(model)?;
    connection.execute(
        "
        INSERT INTO local_models (
            id, model_kind, file_path, file_size_bytes, sha256, version,
            input_format, model_card_id, license_id, license_confirmed_at, imported_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        ",
        params![
            model.id.to_string(),
            serde_json::to_string(&model.model_kind).expect("model kind serializes"),
            model.file_path.to_string_lossy(),
            model.file_size_bytes.to_string(),
            &model.sha256,
            &model.version,
            &model.input_format,
            &model.model_card_id,
            &model.license_id,
            model.license_confirmed_at.to_rfc3339(),
            model.imported_at.to_rfc3339(),
        ],
    )?;
    Ok(())
}

fn validate_registered_model(model: &RegisteredModel) -> Result<(), AuditStoreError> {
    if model.file_path.as_os_str().is_empty()
        || model.file_path.is_absolute()
        || model
            .file_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "managed file path",
            value: model.file_path.display().to_string(),
        });
    }
    if model.file_size_bytes == 0 {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "file size",
            value: "0".to_owned(),
        });
    }
    if model.sha256.len() != 64 || !model.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AuditStoreError::InvalidModelMetadata {
            field: "SHA-256",
            value: model.sha256.clone(),
        });
    }
    for (field, value) in [
        ("model version", model.version.as_str()),
        ("input format", model.input_format.as_str()),
        ("model card identifier", model.model_card_id.as_str()),
        ("license identifier", model.license_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(AuditStoreError::InvalidModelMetadata {
                field,
                value: value.to_owned(),
            });
        }
    }
    Ok(())
}

fn insert_capture_segment(
    connection: &Connection,
    segment: &CaptureSegment,
) -> Result<(), AuditStoreError> {
    connection.execute(
        "
        INSERT INTO capture_segments (
            id, session_id, device_uid, device_name, sample_rate, channels,
            anchor_monotonic_ns, anchor_wall_clock
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        ",
        params![
            segment.id.to_string(),
            segment.session_id.to_string(),
            &segment.device_uid,
            &segment.device_name,
            i64::from(segment.sample_rate),
            i64::from(segment.channels),
            segment.anchor_monotonic_ns.to_string(),
            segment.anchor_wall_clock.to_rfc3339(),
        ],
    )?;
    Ok(())
}

fn validate_capture_gap(gap: &CaptureGap) -> Result<(), AuditStoreError> {
    if gap.ended_at.monotonic_ns < gap.started_at.monotonic_ns {
        return Err(AuditStoreError::InvalidCaptureGapRange);
    }
    Ok(())
}

fn insert_capture_gap(
    connection: &Connection,
    session_id: Uuid,
    gap: &CaptureGap,
) -> Result<(), AuditStoreError> {
    connection.execute(
        "
        INSERT INTO capture_gaps (
            session_id, started_monotonic_ns, started_wall_clock,
            ended_monotonic_ns, ended_wall_clock, reason
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ",
        params![
            session_id.to_string(),
            gap.started_at.monotonic_ns.to_string(),
            gap.started_at.wall_clock.to_rfc3339(),
            gap.ended_at.monotonic_ns.to_string(),
            gap.ended_at.wall_clock.to_rfc3339(),
            serde_json::to_string(&gap.reason).expect("capture gap reason serializes"),
        ],
    )?;
    Ok(())
}

fn insert_inference_gap(
    connection: &Connection,
    gap: &InferenceGap,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    validate_inference_gap(gap)?;
    connection.execute(
        "
        INSERT INTO inference_gaps (
            id, session_id, runtime_id, capture_segment_id, job_id,
            started_monotonic_ns, started_wall_clock,
            ended_monotonic_ns, ended_wall_clock, stage, reason, audit_event_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
        ",
        params![
            gap.id.to_string(),
            gap.session_id.to_string(),
            gap.runtime_id.to_string(),
            gap.capture_segment_id.to_string(),
            gap.job_id.map(|value| value.to_string()),
            gap.started_at.monotonic_ns.to_string(),
            gap.started_at.wall_clock.to_rfc3339(),
            gap.ended_at.monotonic_ns.to_string(),
            gap.ended_at.wall_clock.to_rfc3339(),
            serde_json::to_string(&gap.stage).expect("inference gap stage serializes"),
            serde_json::to_string(&gap.reason).expect("inference gap reason serializes"),
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn validate_capture_segment(segment: &CaptureSegment) -> Result<(), AuditStoreError> {
    if segment.device_uid.trim().is_empty() {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "device uid",
            value: segment.device_uid.clone(),
        });
    }
    if segment.device_name.trim().is_empty() {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "device name",
            value: segment.device_name.clone(),
        });
    }
    if segment.sample_rate == 0 {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "sample rate",
            value: segment.sample_rate.to_string(),
        });
    }
    if segment.channels == 0 {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "channel count",
            value: segment.channels.to_string(),
        });
    }
    Ok(())
}

fn validate_capture_start_bundle<T: Serialize>(
    connection: &Connection,
    session: &CaptureSession,
    segment: &CaptureSegment,
    session_started: &AuditEvent,
    segment_recorded: &AuditEvent,
    input_started: &AuditEvent,
    input_started_payload: &T,
) -> Result<(), AuditStoreError> {
    if segment.session_id != session.id {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "capture segment session ID",
            value: segment.session_id.to_string(),
        });
    }
    validate_capture_segment(segment)?;

    let previous_hash = connection
        .query_row(
            "SELECT hash FROM audit_events ORDER BY sequence DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    validate_capture_start_bundle_event(
        session_started,
        AuditKind::SessionStarted,
        session.id,
        None,
        session.started_monotonic_ns,
        session.started_at,
        session,
        previous_hash.as_deref(),
        "session start",
    )?;
    validate_capture_start_bundle_event(
        segment_recorded,
        AuditKind::CaptureSegmentRecorded,
        session.id,
        None,
        segment.anchor_monotonic_ns,
        segment.anchor_wall_clock,
        segment,
        Some(session_started.hash.as_str()),
        "capture segment",
    )?;
    validate_capture_start_bundle_event(
        input_started,
        AuditKind::CaptureInputStarted,
        session.id,
        None,
        segment.anchor_monotonic_ns,
        segment.anchor_wall_clock,
        input_started_payload,
        Some(segment_recorded.hash.as_str()),
        "capture input start",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_capture_start_bundle_event<T: Serialize>(
    event: &AuditEvent,
    kind: AuditKind,
    session_id: Uuid,
    causation_id: Option<Uuid>,
    monotonic_ns: u64,
    wall_clock: DateTime<Utc>,
    payload: &T,
    previous_hash: Option<&str>,
    label: &'static str,
) -> Result<(), AuditStoreError> {
    if event.kind != kind {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "capture start audit kind",
            value: format!("{label}: {:?}", event.kind),
        });
    }
    if event.run_id != Some(session_id) || event.causation_id != causation_id {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "capture start audit linkage",
            value: format!(
                "{label}: run_id={:?}, causation_id={:?}",
                event.run_id, event.causation_id
            ),
        });
    }
    if event.monotonic_ns != monotonic_ns || event.wall_clock != wall_clock {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "capture start audit timestamp",
            value: format!("{label}: {} at {}", event.monotonic_ns, event.wall_clock),
        });
    }
    if event.previous_hash.as_deref() != previous_hash || !event.verifies() {
        return Err(AuditStoreError::Integrity);
    }
    let payload_matches = event.matches_payload(payload).map_err(|error| {
        AuditStoreError::InvalidCaptureMetadata {
            field: "capture start audit payload",
            value: format!("{label}: {error}"),
        }
    })?;
    if !payload_matches {
        return Err(AuditStoreError::InvalidCaptureMetadata {
            field: "capture start audit payload",
            value: format!("{label}: digest does not match"),
        });
    }
    Ok(())
}

fn parse_capture_monotonic_ns(value: &str) -> Result<u64, AuditStoreError> {
    value
        .parse()
        .map_err(|_| AuditStoreError::InvalidCaptureMetadata {
            field: "monotonic nanoseconds",
            value: value.to_owned(),
        })
}

fn parse_inference_gap_monotonic_ns(value: &str) -> Result<u64, AuditStoreError> {
    value
        .parse()
        .map_err(|_| AuditStoreError::InvalidInferenceGapMetadata {
            field: "monotonic nanoseconds",
            value: value.to_owned(),
        })
}

fn parse_capture_integer<T>(field: &'static str, value: i64) -> Result<T, AuditStoreError>
where
    T: TryFrom<i64>,
{
    value
        .try_into()
        .map_err(|_| AuditStoreError::InvalidCaptureMetadata {
            field,
            value: value.to_string(),
        })
}

fn parse_speaker_u32(field: &'static str, value: i64) -> Result<u32, AuditStoreError> {
    value
        .try_into()
        .map_err(|_| speaker_error(field, value.to_string()))
}

fn parse_speaker_bool(field: &'static str, value: i64) -> Result<bool, AuditStoreError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(speaker_error(field, value.to_string())),
    }
}

fn parse_transcript_monotonic_ns(value: &str) -> Result<u64, AuditStoreError> {
    value
        .parse()
        .map_err(|_| AuditStoreError::InvalidTranscriptMetadata {
            field: "monotonic nanoseconds",
            value: value.to_owned(),
        })
}

fn parse_transcript_revision_number(value: i64) -> Result<u32, AuditStoreError> {
    value
        .try_into()
        .map_err(|_| AuditStoreError::InvalidTranscriptMetadata {
            field: "revision number",
            value: value.to_string(),
        })
}

fn parse_uuid(value: &str) -> Result<Uuid, AuditStoreError> {
    Uuid::parse_str(value).map_err(|_| AuditStoreError::InvalidUuid(value.to_owned()))
}

fn parse_optional_uuid(value: Option<String>) -> Result<Option<Uuid>, AuditStoreError> {
    value.as_deref().map(parse_uuid).transpose()
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>, AuditStoreError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|_| AuditStoreError::InvalidTimestamp(value.to_owned()))
}

fn speaker_error(field: &'static str, value: impl Into<String>) -> AuditStoreError {
    AuditStoreError::InvalidSpeakerMetadata {
        field,
        value: value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CaptureGapReason;
    use crate::audit::AuditKind;
    use crate::inference::{InferenceGap, InferenceGapReason, InferenceGapStage};
    use chrono::Duration;

    fn inference_gap_fixture(session_id: Uuid) -> InferenceGap {
        InferenceGap::new(
            Uuid::new_v4(),
            session_id,
            Uuid::new_v4(),
            Uuid::new_v4(),
            Some(Uuid::new_v4()),
            CapturePoint {
                monotonic_ns: 5_000_000_000,
                wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(5),
            },
            CapturePoint {
                monotonic_ns: 6_000_000_000,
                wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(6),
            },
            InferenceGapStage::JobQueue,
            InferenceGapReason::JobQueueSaturated,
        )
        .unwrap()
    }

    fn inference_gap_event(gap: &InferenceGap, previous_hash: Option<String>) -> AuditEvent {
        AuditEvent::new(
            Some(gap.session_id),
            gap.job_id,
            AuditKind::InferenceGapRecorded,
            gap.ended_at.monotonic_ns,
            gap.ended_at.wall_clock,
            gap,
            previous_hash,
        )
        .unwrap()
    }

    #[test]
    fn persists_reopens_and_verifies_audited_inference_gaps() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-inference-gap-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session_id = Uuid::new_v4();
        let gap = inference_gap_fixture(session_id);
        let event = inference_gap_event(&gap, None);

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store.append_inference_gap_with_audit(&event, &gap).unwrap();
            assert_eq!(
                store.list_inference_gaps(session_id).unwrap(),
                vec![gap.clone()]
            );
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(reopened.list_inference_gaps(session_id).unwrap(), vec![gap]);
        assert!(reopened.verify().unwrap());

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn looks_up_an_inference_gap_with_its_immutable_audit_binding() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);

        store.append_inference_gap_with_audit(&event, &gap).unwrap();

        assert_eq!(
            store.lookup_inference_gap_with_audit(gap.id).unwrap(),
            Some(InferenceGapAuditRecord {
                gap: gap.clone(),
                audit_event: event,
            })
        );
        assert_eq!(
            store
                .lookup_inference_gap_with_audit(Uuid::new_v4())
                .unwrap(),
            None
        );
    }

    #[test]
    fn lookup_rejects_an_inference_gap_bound_to_a_tampered_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);
        store.append_inference_gap_with_audit(&event, &gap).unwrap();
        store
            .connection
            .execute(
                "UPDATE audit_events SET hash = ?1 WHERE id = ?2",
                params!["tampered audit hash", event.id.to_string()],
            )
            .unwrap();
        assert!(store.list().unwrap()[0].matches_payload(&gap).unwrap());
        assert!(matches!(
            store.lookup_inference_gap_with_audit(gap.id),
            Err(AuditStoreError::Integrity)
        ));

        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);
        store.append_inference_gap_with_audit(&event, &gap).unwrap();
        store
            .connection
            .execute(
                "UPDATE audit_events SET previous_hash = ?1 WHERE id = ?2",
                params!["tampered previous hash", event.id.to_string()],
            )
            .unwrap();
        assert!(store.list().unwrap()[0].matches_payload(&gap).unwrap());
        assert!(matches!(
            store.lookup_inference_gap_with_audit(gap.id),
            Err(AuditStoreError::Integrity)
        ));
    }

    #[test]
    fn atomically_persists_and_reopens_a_capture_start_bundle() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-capture-start-bundle-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session = CaptureSession::begin_with_id(
            Uuid::new_v4(),
            2_000_000,
            DateTime::<Utc>::UNIX_EPOCH + Duration::milliseconds(2),
        )
        .unwrap();
        let segment = CaptureSegment::new_with_id(
            Uuid::new_v4(),
            session.id,
            "built-in-mic",
            "Built-in Microphone",
            48_000,
            2,
            session.started_monotonic_ns,
            session.started_at,
        )
        .unwrap();
        let input_payload = serde_json::json!({
            "sessionId": session.id,
            "deviceUid": segment.device_uid.clone(),
            "deviceName": segment.device_name.clone(),
            "sampleRate": segment.sample_rate,
            "channels": segment.channels,
            "anchor": {
                "monotonicNs": segment.anchor_monotonic_ns,
                "wallClock": segment.anchor_wall_clock,
            },
        });
        let session_event = AuditEvent::new(
            Some(session.id),
            None,
            AuditKind::SessionStarted,
            session.started_monotonic_ns,
            session.started_at,
            &session,
            None,
        )
        .unwrap();
        let segment_event = AuditEvent::new(
            Some(session.id),
            None,
            AuditKind::CaptureSegmentRecorded,
            segment.anchor_monotonic_ns,
            segment.anchor_wall_clock,
            &segment,
            Some(session_event.hash.clone()),
        )
        .unwrap();
        let input_event = AuditEvent::new(
            Some(session.id),
            None,
            AuditKind::CaptureInputStarted,
            segment.anchor_monotonic_ns,
            segment.anchor_wall_clock,
            &input_payload,
            Some(segment_event.hash.clone()),
        )
        .unwrap();

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_capture_start_bundle_with_audit(
                    &session,
                    &segment,
                    &session_event,
                    &segment_event,
                    &input_event,
                    &input_payload,
                )
                .unwrap();
            assert_eq!(
                store.list_capture_segments(session.id).unwrap(),
                vec![segment.clone()]
            );
            assert_eq!(
                store
                    .list()
                    .unwrap()
                    .into_iter()
                    .map(|event| event.kind)
                    .collect::<Vec<_>>(),
                vec![
                    AuditKind::SessionStarted,
                    AuditKind::CaptureSegmentRecorded,
                    AuditKind::CaptureInputStarted,
                ]
            );
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(
            reopened.list_capture_segments(session.id).unwrap(),
            vec![segment]
        );
        assert!(reopened.verify().unwrap());
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn rejects_inference_gaps_without_their_matching_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());

        let wrong_kind = AuditEvent::new(
            Some(gap.session_id),
            gap.job_id,
            AuditKind::CaptureGapRecorded,
            gap.ended_at.monotonic_ns,
            gap.ended_at.wall_clock,
            &gap,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_inference_gap_with_audit(&wrong_kind, &gap),
            Err(AuditStoreError::InvalidInferenceGapMetadata {
                field: "audit event kind",
                ..
            })
        ));

        let wrong_payload = AuditEvent::new(
            Some(gap.session_id),
            gap.job_id,
            AuditKind::InferenceGapRecorded,
            gap.ended_at.monotonic_ns,
            gap.ended_at.wall_clock,
            &serde_json::json!({ "gap": "different" }),
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_inference_gap_with_audit(&wrong_payload, &gap),
            Err(AuditStoreError::InvalidInferenceGapMetadata {
                field: "audit event payload",
                ..
            })
        ));

        assert!(store.list().unwrap().is_empty());
        assert!(store
            .list_inference_gaps(gap.session_id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn verification_rejects_tampered_or_missing_inference_gap_bindings() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);
        store.append_inference_gap_with_audit(&event, &gap).unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute_batch("DROP TRIGGER inference_gaps_are_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE inference_gaps SET reason = ?1 WHERE id = ?2",
                params![
                    serde_json::to_string(&InferenceGapReason::EngineFailed).unwrap(),
                    gap.id.to_string(),
                ],
            )
            .unwrap();
        assert!(!store.verify().unwrap());

        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);
        store.append_inference_gap_with_audit(&event, &gap).unwrap();
        store
            .connection
            .execute_batch("DROP TRIGGER inference_gaps_are_immutable_delete;")
            .unwrap();
        store
            .connection
            .execute(
                "DELETE FROM inference_gaps WHERE id = ?1",
                params![gap.id.to_string()],
            )
            .unwrap();
        assert!(!store.verify().unwrap());
    }

    #[test]
    fn prevents_duplicate_inference_gap_audit_bindings() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let gap = inference_gap_fixture(Uuid::new_v4());
        let event = inference_gap_event(&gap, None);
        store.append_inference_gap_with_audit(&event, &gap).unwrap();

        let duplicate = InferenceGap {
            id: Uuid::new_v4(),
            ..gap.clone()
        };
        assert!(store
            .connection
            .execute(
                "
                INSERT INTO inference_gaps (
                    id, session_id, runtime_id, capture_segment_id, job_id,
                    started_monotonic_ns, started_wall_clock,
                    ended_monotonic_ns, ended_wall_clock, stage, reason, audit_event_id
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                ",
                params![
                    duplicate.id.to_string(),
                    duplicate.session_id.to_string(),
                    duplicate.runtime_id.to_string(),
                    duplicate.capture_segment_id.to_string(),
                    duplicate.job_id.map(|value| value.to_string()),
                    duplicate.started_at.monotonic_ns.to_string(),
                    duplicate.started_at.wall_clock.to_rfc3339(),
                    duplicate.ended_at.monotonic_ns.to_string(),
                    duplicate.ended_at.wall_clock.to_rfc3339(),
                    serde_json::to_string(&duplicate.stage).unwrap(),
                    serde_json::to_string(&duplicate.reason).unwrap(),
                    event.id.to_string(),
                ],
            )
            .is_err());
        assert!(store.verify().unwrap());
    }

    #[test]
    fn persists_a_verifiable_audit_chain() {
        let store = AuditStore::open_in_memory().unwrap();
        let event = AuditEvent::new(
            None,
            None,
            AuditKind::ActionDenied,
            42,
            Utc::now(),
            &serde_json::json!({ "reason": "denied_by_default" }),
            None,
        )
        .unwrap();

        store.append(&event).unwrap();

        assert_eq!(store.list().unwrap(), vec![event]);
        assert!(store.verify().unwrap());
    }

    #[test]
    fn rejects_an_event_that_does_not_extend_the_durable_chain() {
        let store = AuditStore::open_in_memory().unwrap();
        let first = AuditEvent::new(
            None,
            None,
            AuditKind::ActionDenied,
            1,
            Utc::now(),
            &serde_json::json!({ "reason": "first" }),
            None,
        )
        .unwrap();
        store.append(&first).unwrap();

        let divergent = AuditEvent::new(
            None,
            None,
            AuditKind::ActionDenied,
            2,
            Utc::now(),
            &serde_json::json!({ "reason": "divergent" }),
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append(&divergent),
            Err(AuditStoreError::Integrity)
        ));
        assert_eq!(store.list().unwrap(), vec![first]);
        assert!(store.verify().unwrap());
    }

    #[test]
    fn reopens_capture_segments_and_gaps_without_breaking_the_audit_chain() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-capture-metadata-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session_id = Uuid::new_v4();
        let anchor_wall_clock = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(5);
        let segment = CaptureSegment::new(
            session_id,
            "built-in-mic",
            "Built-in Microphone",
            48_000,
            2,
            5_000_000_000,
            anchor_wall_clock,
        )
        .unwrap();
        let gap = CaptureGap {
            started_at: CapturePoint {
                monotonic_ns: 8_000_000_000,
                wall_clock: anchor_wall_clock + Duration::seconds(3),
            },
            ended_at: CapturePoint {
                monotonic_ns: 9_500_000_000,
                wall_clock: anchor_wall_clock + Duration::milliseconds(4_500),
            },
            reason: CaptureGapReason::QueueOverrun,
        };

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            let segment_event = AuditEvent::new(
                Some(session_id),
                None,
                AuditKind::CaptureSegmentRecorded,
                segment.anchor_monotonic_ns,
                segment.anchor_wall_clock,
                &segment,
                None,
            )
            .unwrap();
            store
                .append_capture_segment_with_audit(&segment_event, &segment)
                .unwrap();

            let gap_event = AuditEvent::new(
                Some(session_id),
                None,
                AuditKind::CaptureGapRecorded,
                gap.ended_at.monotonic_ns,
                gap.ended_at.wall_clock,
                &gap,
                Some(segment_event.hash.clone()),
            )
            .unwrap();
            store
                .append_capture_gap_with_audit(&gap_event, session_id, &gap)
                .unwrap();

            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(
            reopened.list_capture_segments(session_id).unwrap(),
            vec![segment]
        );
        assert_eq!(reopened.list_capture_gaps(session_id).unwrap(), vec![gap]);
        let audit_events = reopened.list().unwrap();
        assert_eq!(audit_events.len(), 2);
        assert_eq!(audit_events[0].kind, AuditKind::CaptureSegmentRecorded);
        assert_eq!(audit_events[1].kind, AuditKind::CaptureGapRecorded);
        assert!(AuditTrail::from_events(audit_events).verify());

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn rejects_inverted_capture_gap_ranges() {
        let store = AuditStore::open_in_memory().unwrap();
        let gap = CaptureGap {
            started_at: CapturePoint {
                monotonic_ns: 20,
                wall_clock: Utc::now(),
            },
            ended_at: CapturePoint {
                monotonic_ns: 10,
                wall_clock: Utc::now(),
            },
            reason: CaptureGapReason::SystemSleep,
        };

        assert!(matches!(
            store.append_capture_gap(Uuid::new_v4(), &gap),
            Err(AuditStoreError::InvalidCaptureGapRange)
        ));
    }

    #[test]
    fn rolls_back_the_audit_event_when_capture_segment_metadata_is_rejected() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let anchor_wall_clock = Utc::now();
        let segment = CaptureSegment::new(
            session_id,
            "built-in-mic",
            "Built-in Microphone",
            48_000,
            2,
            5_000_000_000,
            anchor_wall_clock,
        )
        .unwrap();
        let first_event = AuditEvent::new(
            Some(session_id),
            None,
            AuditKind::CaptureSegmentRecorded,
            segment.anchor_monotonic_ns,
            segment.anchor_wall_clock,
            &segment,
            None,
        )
        .unwrap();
        store
            .append_capture_segment_with_audit(&first_event, &segment)
            .unwrap();

        let conflicting_event = AuditEvent::new(
            Some(session_id),
            None,
            AuditKind::CaptureSegmentRecorded,
            segment.anchor_monotonic_ns + 1,
            segment.anchor_wall_clock,
            &segment,
            Some(first_event.hash.clone()),
        )
        .unwrap();
        assert!(store
            .append_capture_segment_with_audit(&conflicting_event, &segment)
            .is_err());

        assert_eq!(store.list().unwrap(), vec![first_event]);
        assert_eq!(
            store.list_capture_segments(session_id).unwrap(),
            vec![segment]
        );
        assert!(store.verify().unwrap());
    }

    fn transcript_fixture(session_id: Uuid) -> TranscriptRevision {
        let start = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(20);
        TranscriptRevision::original(
            session_id,
            crate::domain::TranscriptTiming::new(
                20_000_000_000,
                21_200_000_000,
                start,
                start + Duration::milliseconds(1_200),
            )
            .unwrap(),
            Some("speaker-1".to_owned()),
            "local fixture original",
            true,
            TranscriptSource::Synthetic,
            Some(
                TranscriptModelProvenance::new(
                    "whisper.cpp",
                    "ggml-small",
                    "1.7.4",
                    Some("a".repeat(64)),
                )
                .unwrap(),
            ),
            Some(0.81),
        )
        .unwrap()
    }

    fn transcript_event(
        revision: &TranscriptRevision,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        AuditEvent::new(
            Some(revision.session_id),
            revision.parent_revision_id,
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            revision,
            previous_hash,
        )
        .unwrap()
    }

    fn speaker_creation(
        cluster: &SpeakerClusterRecord,
        initial_label: &SpeakerClusterLabelRevision,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        let payload =
            SpeakerClusterCreatedAuditPayload::new(cluster.clone(), initial_label.clone()).unwrap();
        AuditEvent::new(
            Some(cluster.session_id),
            None,
            AuditKind::SpeakerClusterCreated,
            1,
            DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(1),
            &payload,
            previous_hash,
        )
        .unwrap()
    }

    fn speaker_label_event(
        cluster: &SpeakerClusterRecord,
        revision: &SpeakerClusterLabelRevision,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        AuditEvent::new(
            Some(cluster.session_id),
            revision.parent_revision_id,
            AuditKind::SpeakerClusterLabelRevisionRecorded,
            2,
            DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(2),
            revision,
            previous_hash,
        )
        .unwrap()
    }

    fn speaker_alias_event(
        cluster: &SpeakerClusterRecord,
        revision: &SpeakerClusterAliasRevision,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        AuditEvent::new(
            Some(cluster.session_id),
            revision.parent_revision_id,
            AuditKind::SpeakerClusterAliasRevisionRecorded,
            3,
            DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(3),
            revision,
            previous_hash,
        )
        .unwrap()
    }

    fn speaker_reassignment_event(
        revision: &TranscriptRevision,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        AuditEvent::new(
            Some(revision.session_id),
            revision.parent_revision_id,
            AuditKind::TranscriptSpeakerReassigned,
            revision.capture_end_ns + 1,
            revision.wall_clock_end + Duration::seconds(1),
            revision,
            previous_hash,
        )
        .unwrap()
    }

    fn asr_final_fixture(
        session_id: Uuid,
        utterance_key: &str,
        emission_revision: u32,
        emission_payload_sha256: impl Into<String>,
    ) -> (
        TranscriptRevision,
        AsrFinalIdempotencyKey,
        AsrFinalIdempotencyBinding,
    ) {
        let key = AsrFinalIdempotencyKey::new(session_id, utterance_key, emission_revision)
            .expect("fixture ASR key is valid");
        let mut revision = transcript_fixture(session_id);
        revision.source = TranscriptSource::LocalInference;
        let logical_span_id = logical_span_id_for_asr_utterance_digest(
            session_id,
            &key.opaque_utterance_key_sha256(),
        );
        revision.id = logical_span_id;
        revision.logical_span_id = logical_span_id;
        let idempotency = AsrFinalIdempotencyBinding::new(&key, &revision, emission_payload_sha256)
            .expect("fixture ASR idempotency binding is valid");

        (revision, key, idempotency)
    }

    fn asr_final_event(
        revision: &TranscriptRevision,
        idempotency: &AsrFinalIdempotencyBinding,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        let payload = AsrFinalAuditPayload::new(revision, idempotency);
        AuditEvent::new(
            Some(revision.session_id),
            revision.parent_revision_id,
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            &payload,
            previous_hash,
        )
        .unwrap()
    }

    #[test]
    fn atomically_binds_a_final_asr_emission_to_its_revision_across_reopen() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-asr-final-idempotency-{}.sqlite3",
            Uuid::new_v4()
        ));
        let payload_sha256 = "d".repeat(64);
        let (revision, key, idempotency) = asr_final_fixture(
            Uuid::new_v4(),
            "fixture-utterance-1",
            2,
            payload_sha256.clone(),
        );
        let event = asr_final_event(&revision, &idempotency, None);

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_asr_final_transcript_revision_with_audit(&event, &revision, &idempotency)
                .unwrap();
            assert_eq!(
                store.lookup_asr_final_idempotency(&key).unwrap(),
                Some(AsrFinalIdempotencyRecord {
                    revision_id: revision.id,
                    emission_payload_sha256: payload_sha256.clone(),
                })
            );
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(
            reopened.lookup_asr_final_idempotency(&key).unwrap(),
            Some(AsrFinalIdempotencyRecord {
                revision_id: revision.id,
                emission_payload_sha256: payload_sha256,
            })
        );
        assert_eq!(
            reopened
                .list_transcript_revisions(revision.session_id)
                .unwrap(),
            vec![revision]
        );
        assert!(reopened.verify().unwrap());

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn verification_rejects_an_asr_final_payload_digest_tampered_after_its_audit_event() {
        let payload_sha256 = "d".repeat(64);
        let (revision, _key, idempotency) = asr_final_fixture(
            Uuid::new_v4(),
            "fixture-utterance-payload-tamper",
            1,
            payload_sha256,
        );
        let event = asr_final_event(&revision, &idempotency, None);
        let mut store = AuditStore::open_in_memory().unwrap();
        store
            .append_asr_final_transcript_revision_with_audit(&event, &revision, &idempotency)
            .unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute_batch("DROP TRIGGER asr_final_idempotency_is_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE asr_final_idempotency SET emission_payload_sha256 = ?1 WHERE revision_id = ?2",
                params!["e".repeat(64), revision.id.to_string()],
            )
            .unwrap();

        assert!(!store.verify().unwrap());
    }

    #[test]
    fn verification_rejects_an_asr_final_missing_its_idempotency_binding() {
        let (revision, _key, idempotency) = asr_final_fixture(
            Uuid::new_v4(),
            "fixture-utterance-binding-delete",
            1,
            "d".repeat(64),
        );
        let event = asr_final_event(&revision, &idempotency, None);
        let mut store = AuditStore::open_in_memory().unwrap();
        store
            .append_asr_final_transcript_revision_with_audit(&event, &revision, &idempotency)
            .unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute_batch("DROP TRIGGER asr_final_idempotency_is_immutable_delete;")
            .unwrap();
        store
            .connection
            .execute(
                "DELETE FROM asr_final_idempotency WHERE revision_id = ?1",
                params![revision.id.to_string()],
            )
            .unwrap();

        assert!(!store.verify().unwrap());
    }

    #[test]
    fn verification_rejects_an_asr_final_key_rebound_to_a_different_logical_span() {
        let (revision, _key, idempotency) = asr_final_fixture(
            Uuid::new_v4(),
            "fixture-utterance-key-tamper",
            1,
            "d".repeat(64),
        );
        let event = asr_final_event(&revision, &idempotency, None);
        let mut store = AuditStore::open_in_memory().unwrap();
        store
            .append_asr_final_transcript_revision_with_audit(&event, &revision, &idempotency)
            .unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute_batch("DROP TRIGGER asr_final_idempotency_is_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE asr_final_idempotency SET utterance_key_sha256 = ?1 WHERE revision_id = ?2",
                params!["e".repeat(64), revision.id.to_string()],
            )
            .unwrap();

        assert!(!store.verify().unwrap());
    }

    fn local_model_fixture() -> RegisteredModel {
        RegisteredModel {
            id: Uuid::new_v4(),
            model_kind: LocalModelKind::SpeechRecognition,
            file_path: "models/fixture.model".into(),
            file_size_bytes: 8_192,
            sha256: "b".repeat(64),
            version: "1.7.4".to_owned(),
            input_format: "gguf".to_owned(),
            model_card_id: "example/model-card".to_owned(),
            license_id: "mit".to_owned(),
            license_confirmed_at: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(5),
            imported_at: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(10),
        }
    }

    fn local_model_event(model: &RegisteredModel, previous_hash: Option<String>) -> AuditEvent {
        AuditEvent::new(
            None,
            None,
            AuditKind::LocalModelImported,
            10_000_000_000,
            model.imported_at,
            model,
            previous_hash,
        )
        .unwrap()
    }

    #[test]
    fn rejects_local_models_without_a_matching_bound_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let model = local_model_fixture();

        let wrong_kind = AuditEvent::new(
            None,
            None,
            AuditKind::TranscriptRevisionRecorded,
            10_000_000_000,
            model.imported_at,
            &model,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_local_model_with_audit(&wrong_kind, &model),
            Err(AuditStoreError::InvalidModelMetadata {
                field: "audit event kind",
                ..
            })
        ));

        let mismatched = RegisteredModel {
            version: "2.0.0".to_owned(),
            ..model.clone()
        };
        let wrong_payload = local_model_event(&mismatched, None);
        assert!(matches!(
            store.append_local_model_with_audit(&wrong_payload, &model),
            Err(AuditStoreError::InvalidModelMetadata {
                field: "audit event payload",
                ..
            })
        ));

        let wrong_session = AuditEvent::new(
            Some(Uuid::new_v4()),
            None,
            AuditKind::LocalModelImported,
            10_000_000_000,
            model.imported_at,
            &model,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_local_model_with_audit(&wrong_session, &model),
            Err(AuditStoreError::InvalidModelMetadata {
                field: "audit event session linkage",
                ..
            })
        ));

        let wrong_causation = AuditEvent::new(
            None,
            Some(Uuid::new_v4()),
            AuditKind::LocalModelImported,
            10_000_000_000,
            model.imported_at,
            &model,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_local_model_with_audit(&wrong_causation, &model),
            Err(AuditStoreError::InvalidModelMetadata {
                field: "audit event causation linkage",
                ..
            })
        ));

        assert!(store.list().unwrap().is_empty());
        assert!(store.list_local_models().unwrap().is_empty());

        let valid_event = local_model_event(&model, None);
        store
            .append_local_model_with_audit(&valid_event, &model)
            .unwrap();
        assert_eq!(store.list().unwrap(), vec![valid_event]);
        assert_eq!(store.list_local_models().unwrap(), vec![model]);
        assert!(store.verify().unwrap());
    }

    #[test]
    fn rejects_transcript_records_without_a_matching_bound_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let revision = transcript_fixture(Uuid::new_v4());

        let wrong_kind = AuditEvent::new(
            Some(revision.session_id),
            None,
            AuditKind::TranscriptRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            &revision,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_transcript_revision_with_audit(&wrong_kind, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "audit event kind",
                ..
            })
        ));

        let wrong_payload = AuditEvent::new(
            Some(revision.session_id),
            None,
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            &serde_json::json!({ "revision": "different" }),
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_transcript_revision_with_audit(&wrong_payload, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "audit event payload",
                ..
            })
        ));

        let wrong_session = AuditEvent::new(
            Some(Uuid::new_v4()),
            None,
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            &revision,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_transcript_revision_with_audit(&wrong_session, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "audit event session linkage",
                ..
            })
        ));

        let wrong_causation = AuditEvent::new(
            Some(revision.session_id),
            Some(Uuid::new_v4()),
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns,
            revision.wall_clock_end,
            &revision,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_transcript_revision_with_audit(&wrong_causation, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "audit event causation linkage",
                ..
            })
        ));

        let wrong_capture_endpoint = AuditEvent::new(
            Some(revision.session_id),
            None,
            AuditKind::TranscriptRevisionRecorded,
            revision.capture_end_ns + 1,
            revision.wall_clock_end,
            &revision,
            None,
        )
        .unwrap();
        assert!(matches!(
            store.append_transcript_revision_with_audit(&wrong_capture_endpoint, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "audit event capture endpoint",
                ..
            })
        ));

        assert!(store.list().unwrap().is_empty());
        assert!(store
            .list_transcript_revisions(revision.session_id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn rejects_unbound_local_inference_revisions_on_write_and_verification() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let mut revision = transcript_fixture(Uuid::new_v4());
        revision.source = TranscriptSource::LocalInference;
        let event = transcript_event(&revision, None);

        assert!(matches!(
            store.append_transcript_revision_with_audit(&event, &revision),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "local inference audit binding",
                ..
            })
        ));
        assert!(store.list().unwrap().is_empty());
        assert!(store
            .list_transcript_revisions(revision.session_id)
            .unwrap()
            .is_empty());

        // Simulate a legacy or directly-tampered SQLite row. Its generic
        // audit payload is valid, but a local ASR record without its durable
        // idempotency binding must fail verification after reopening.
        insert_audit_event(&store.connection, &event).unwrap();
        insert_transcript_revision(&store.connection, &revision).unwrap();
        assert!(!store.verify().unwrap());
    }

    #[test]
    fn verification_rejects_a_transcript_tampered_after_its_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let revision = transcript_fixture(Uuid::new_v4());
        let event = transcript_event(&revision, None);
        store
            .append_transcript_revision_with_audit(&event, &revision)
            .unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute_batch("DROP TRIGGER transcript_revisions_are_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE transcript_revisions SET text = ?1 WHERE id = ?2",
                params!["tampered local text", revision.id.to_string()],
            )
            .unwrap();

        assert!(!store.verify().unwrap());
    }

    #[test]
    fn verification_rejects_a_model_tampered_after_its_audit_event() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let model = local_model_fixture();
        let event = local_model_event(&model, None);
        store.append_local_model_with_audit(&event, &model).unwrap();
        assert!(store.verify().unwrap());

        store
            .connection
            .execute(
                "UPDATE local_models SET version = ?1 WHERE id = ?2",
                params!["tampered-version", model.id.to_string()],
            )
            .unwrap();

        assert!(!store.verify().unwrap());
    }

    #[test]
    fn persists_revisioned_transcripts_and_searches_the_local_fts_projection() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-transcript-revisions-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session_id = Uuid::new_v4();
        let original = transcript_fixture(session_id);
        let revised = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            original.speaker_cluster_id.clone(),
            "local fixture finalized",
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            None,
        )
        .unwrap();

        let original_event = transcript_event(&original, None);
        let revised_event = transcript_event(&revised, Some(original_event.hash.clone()));
        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_transcript_revision_with_audit(&original_event, &original)
                .unwrap();
            store
                .append_transcript_revision_with_audit(&revised_event, &revised)
                .unwrap();

            assert_eq!(
                store.list_transcript_revisions(session_id).unwrap(),
                vec![original.clone(), revised.clone()]
            );
            assert_eq!(
                store
                    .search_transcript_revisions(Some(session_id), "fixture")
                    .unwrap(),
                vec![original.clone(), revised.clone()]
            );
            assert!(store
                .search_transcript_revisions(Some(session_id), "")
                .unwrap()
                .is_empty());
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(
            reopened.list_transcript_revisions(session_id).unwrap(),
            vec![original.clone(), revised.clone()]
        );
        assert_eq!(
            reopened
                .search_transcript_revisions(None, "fixture")
                .unwrap(),
            vec![original, revised]
        );
        assert!(reopened.verify().unwrap());

        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn rolls_back_audit_when_a_transcript_parent_is_missing() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let original = transcript_fixture(Uuid::new_v4());
        let original_event = transcript_event(&original, None);
        store
            .append_transcript_revision_with_audit(&original_event, &original)
            .unwrap();

        let mut invalid = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            original.speaker_cluster_id.clone(),
            "local fixture unavailable parent",
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            None,
        )
        .unwrap();
        invalid.parent_revision_id = Some(Uuid::new_v4());
        let invalid_event = transcript_event(&invalid, Some(original_event.hash.clone()));

        assert!(matches!(
            store.append_transcript_revision_with_audit(&invalid_event, &invalid),
            Err(AuditStoreError::MissingTranscriptParent(_))
        ));
        assert_eq!(store.list().unwrap(), vec![original_event]);
        assert_eq!(
            store
                .list_transcript_revisions(original.session_id)
                .unwrap(),
            vec![original]
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn keeps_nonfinal_transcript_output_out_of_durable_storage() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let mut transient = transcript_fixture(Uuid::new_v4());
        transient.is_final = false;
        let event = transcript_event(&transient, None);

        assert!(matches!(
            store.append_transcript_revision_with_audit(&event, &transient),
            Err(AuditStoreError::NonFinalTranscript)
        ));
        assert!(store.list().unwrap().is_empty());
        assert!(store
            .list_transcript_revisions(transient.session_id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn database_rejects_overwriting_an_immutable_transcript_revision() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let original = transcript_fixture(Uuid::new_v4());
        let event = transcript_event(&original, None);
        store
            .append_transcript_revision_with_audit(&event, &original)
            .unwrap();

        let error = store
            .connection
            .execute(
                "UPDATE transcript_revisions SET text = ?1 WHERE id = ?2",
                params!["rewritten", original.id.to_string()],
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("transcript revisions are immutable"));
        assert_eq!(
            store
                .list_transcript_revisions(original.session_id)
                .unwrap(),
            vec![original]
        );
    }

    #[test]
    fn rejects_parallel_revisions_at_the_same_logical_version() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let original = transcript_fixture(Uuid::new_v4());
        let first = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            original.speaker_cluster_id.clone(),
            "first linear revision",
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            None,
        )
        .unwrap();
        let parallel = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            original.speaker_cluster_id.clone(),
            "parallel revision",
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            None,
        )
        .unwrap();

        let original_event = transcript_event(&original, None);
        let first_event = transcript_event(&first, Some(original_event.hash.clone()));
        let parallel_event = transcript_event(&parallel, Some(first_event.hash.clone()));
        store
            .append_transcript_revision_with_audit(&original_event, &original)
            .unwrap();
        store
            .append_transcript_revision_with_audit(&first_event, &first)
            .unwrap();
        assert!(store
            .append_transcript_revision_with_audit(&parallel_event, &parallel)
            .is_err());
        assert_eq!(store.list().unwrap(), vec![original_event, first_event]);
        assert_eq!(
            store
                .list_transcript_revisions(original.session_id)
                .unwrap(),
            vec![original, first]
        );
    }

    #[test]
    fn persists_reopens_and_projects_audited_speaker_catalog_revisions() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-speaker-catalog-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session_id = Uuid::new_v4();
        let first = SpeakerClusterRecord::new(session_id, 1).unwrap();
        let first_initial = SpeakerClusterLabelRevision::initial_generated(&first).unwrap();
        let first_event = speaker_creation(&first, &first_initial, None);
        let second = SpeakerClusterRecord::new(session_id, 2).unwrap();
        let second_initial = SpeakerClusterLabelRevision::initial_generated(&second).unwrap();
        let second_event =
            speaker_creation(&second, &second_initial, Some(first_event.hash.clone()));
        let renamed = SpeakerClusterLabelRevision::revision_of(&first_initial, "主持人").unwrap();
        let rename_event = speaker_label_event(&first, &renamed, Some(second_event.hash.clone()));
        let alias =
            SpeakerClusterAliasRevision::aliased_to(first.id.clone(), second.id.clone()).unwrap();
        let alias_event = speaker_alias_event(&first, &alias, Some(rename_event.hash.clone()));
        let mut transcript = transcript_fixture(session_id);
        transcript.speaker_cluster_id = Some(first.id.clone());
        let transcript_event = transcript_event(&transcript, Some(alias_event.hash.clone()));

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_speaker_cluster_with_audit(&first_event, &first, &first_initial)
                .unwrap();
            store
                .append_speaker_cluster_with_audit(&second_event, &second, &second_initial)
                .unwrap();
            store
                .append_speaker_cluster_label_revision_with_audit(&rename_event, &renamed)
                .unwrap();
            store
                .append_speaker_cluster_alias_revision_with_audit(&alias_event, &alias)
                .unwrap();
            store
                .append_transcript_revision_with_audit(&transcript_event, &transcript)
                .unwrap();

            let clusters = store.list_speaker_clusters(session_id).unwrap();
            let projected = clusters
                .iter()
                .find(|cluster| cluster.id == first.id)
                .unwrap();
            assert_eq!(projected.label, "主持人");
            assert!(projected.is_user_named);
            assert_eq!(projected.label_revision, 2);
            assert_eq!(projected.alias_revision, 1);
            assert_eq!(projected.merged_into_cluster_id, Some(second.id.clone()));
            assert_eq!(projected.canonical_cluster_id, second.id);
            assert_eq!(projected.span_count, 1);
            assert_eq!(
                store
                    .get_latest_speaker_cluster_label_revision(&first.id)
                    .unwrap(),
                Some(renamed.clone())
            );
            assert_eq!(
                store.get_speaker_cluster_record(&first.id).unwrap(),
                Some(first.clone())
            );
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(reopened.list_speaker_clusters(session_id).unwrap().len(), 2);
        assert!(reopened.verify().unwrap());
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn rejects_invalid_speaker_audit_bindings_and_label_parent_gaps() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let cluster = SpeakerClusterRecord::new(session_id, 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let created = speaker_creation(&cluster, &initial, None);
        store
            .append_speaker_cluster_with_audit(&created, &cluster, &initial)
            .unwrap();
        let renamed = SpeakerClusterLabelRevision::revision_of(&initial, "发言人").unwrap();

        let wrong_kind = AuditEvent::new(
            Some(session_id),
            Some(initial.id),
            AuditKind::SpeakerClusterAliasRevisionRecorded,
            2,
            Utc::now(),
            &renamed,
            Some(created.hash.clone()),
        )
        .unwrap();
        assert!(matches!(
            store.append_speaker_cluster_label_revision_with_audit(&wrong_kind, &renamed),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "audit event kind",
                ..
            })
        ));

        let wrong_run = AuditEvent::new(
            Some(Uuid::new_v4()),
            Some(initial.id),
            AuditKind::SpeakerClusterLabelRevisionRecorded,
            2,
            Utc::now(),
            &renamed,
            Some(created.hash.clone()),
        )
        .unwrap();
        assert!(matches!(
            store.append_speaker_cluster_label_revision_with_audit(&wrong_run, &renamed),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "audit event session linkage",
                ..
            })
        ));

        let wrong_causation = AuditEvent::new(
            Some(session_id),
            None,
            AuditKind::SpeakerClusterLabelRevisionRecorded,
            2,
            Utc::now(),
            &renamed,
            Some(created.hash.clone()),
        )
        .unwrap();
        assert!(matches!(
            store.append_speaker_cluster_label_revision_with_audit(&wrong_causation, &renamed),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "audit event causation linkage",
                ..
            })
        ));

        let wrong_payload = AuditEvent::new(
            Some(session_id),
            Some(initial.id),
            AuditKind::SpeakerClusterLabelRevisionRecorded,
            2,
            Utc::now(),
            &serde_json::json!({ "revision": "wrong" }),
            Some(created.hash.clone()),
        )
        .unwrap();
        assert!(matches!(
            store.append_speaker_cluster_label_revision_with_audit(&wrong_payload, &renamed),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "audit event payload",
                ..
            })
        ));

        let skipped = SpeakerClusterLabelRevision::new_with_id(
            Uuid::new_v4(),
            cluster.id.clone(),
            Some(initial.id),
            3,
            "跳过",
            true,
        )
        .unwrap();
        let skipped_event = speaker_label_event(&cluster, &skipped, Some(created.hash.clone()));
        assert!(matches!(
            store.append_speaker_cluster_label_revision_with_audit(&skipped_event, &skipped),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "speaker label parent",
                ..
            })
        ));
        assert_eq!(store.list().unwrap(), vec![created]);
        assert!(store.verify().unwrap());
    }

    #[test]
    fn rejects_cross_session_and_cyclic_speaker_aliases() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let first = SpeakerClusterRecord::new(session_id, 1).unwrap();
        let first_initial = SpeakerClusterLabelRevision::initial_generated(&first).unwrap();
        let first_event = speaker_creation(&first, &first_initial, None);
        let second = SpeakerClusterRecord::new(session_id, 2).unwrap();
        let second_initial = SpeakerClusterLabelRevision::initial_generated(&second).unwrap();
        let second_event =
            speaker_creation(&second, &second_initial, Some(first_event.hash.clone()));
        let foreign = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let foreign_initial = SpeakerClusterLabelRevision::initial_generated(&foreign).unwrap();
        let foreign_event =
            speaker_creation(&foreign, &foreign_initial, Some(second_event.hash.clone()));
        store
            .append_speaker_cluster_with_audit(&first_event, &first, &first_initial)
            .unwrap();
        store
            .append_speaker_cluster_with_audit(&second_event, &second, &second_initial)
            .unwrap();
        store
            .append_speaker_cluster_with_audit(&foreign_event, &foreign, &foreign_initial)
            .unwrap();

        let alias =
            SpeakerClusterAliasRevision::aliased_to(first.id.clone(), second.id.clone()).unwrap();
        let alias_event = speaker_alias_event(&first, &alias, Some(foreign_event.hash.clone()));
        store
            .append_speaker_cluster_alias_revision_with_audit(&alias_event, &alias)
            .unwrap();

        let cycle =
            SpeakerClusterAliasRevision::aliased_to(second.id.clone(), first.id.clone()).unwrap();
        let cycle_event = speaker_alias_event(&second, &cycle, Some(alias_event.hash.clone()));
        assert!(matches!(
            store.append_speaker_cluster_alias_revision_with_audit(&cycle_event, &cycle),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "speaker alias cycle",
                ..
            })
        ));

        let cross_session =
            SpeakerClusterAliasRevision::revision_of(&alias, Some(foreign.id.clone())).unwrap();
        let cross_session_event =
            speaker_alias_event(&first, &cross_session, Some(alias_event.hash.clone()));
        assert!(matches!(
            store.append_speaker_cluster_alias_revision_with_audit(
                &cross_session_event,
                &cross_session,
            ),
            Err(AuditStoreError::InvalidSpeakerMetadata {
                field: "speaker alias session",
                ..
            })
        ));
        assert!(
            SpeakerClusterAliasRevision::aliased_to(first.id.clone(), first.id.clone()).is_err()
        );
        assert!(store.verify().unwrap());
    }

    #[test]
    fn enforces_speaker_immutability_duplicate_bindings_and_verification() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let cluster = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let event = speaker_creation(&cluster, &initial, None);
        store
            .append_speaker_cluster_with_audit(&event, &cluster, &initial)
            .unwrap();
        assert!(store
            .connection
            .execute(
                "UPDATE speaker_cluster_label_revisions SET label = ?1 WHERE id = ?2",
                params!["rewritten", initial.id.to_string()],
            )
            .is_err());
        assert!(store
            .connection
            .execute(
                "
                INSERT INTO speaker_cluster_label_revisions (
                    id, speaker_cluster_id, parent_revision_id, revision, label, is_user_named,
                    audit_event_id
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ",
                params![
                    Uuid::new_v4().to_string(),
                    &cluster.id,
                    initial.id.to_string(),
                    2_i64,
                    "duplicate binding",
                    1_i64,
                    event.id.to_string(),
                ],
            )
            .is_err());
        store
            .connection
            .execute_batch("DROP TRIGGER speaker_cluster_label_revisions_are_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE speaker_cluster_label_revisions SET label = ?1 WHERE id = ?2",
                params!["rewritten", initial.id.to_string()],
            )
            .unwrap();
        assert!(!store.verify().unwrap());

        let mut store = AuditStore::open_in_memory().unwrap();
        let cluster = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&cluster).unwrap();
        let event = speaker_creation(&cluster, &initial, None);
        store
            .append_speaker_cluster_with_audit(&event, &cluster, &initial)
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE audit_events SET hash = ?1 WHERE id = ?2",
                params!["tampered audit hash", event.id.to_string()],
            )
            .unwrap();
        assert!(!store.verify().unwrap());
    }

    #[test]
    fn keeps_legacy_speaker_strings_and_binds_edit_time_speaker_reassignments() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let original = transcript_fixture(Uuid::new_v4());
        let original_event = transcript_event(&original, None);
        store
            .append_transcript_revision_with_audit(&original_event, &original)
            .unwrap();
        assert!(store
            .list_speaker_clusters(original.session_id)
            .unwrap()
            .is_empty());
        assert!(store.verify().unwrap());

        let reassigned = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            None,
            original.text.clone(),
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            original.confidence,
        )
        .unwrap();
        let generic_event = transcript_event(&reassigned, Some(original_event.hash.clone()));
        assert!(matches!(
            store.append_transcript_revision_with_audit(&generic_event, &reassigned),
            Err(AuditStoreError::InvalidTranscriptMetadata {
                field: "speaker reassignment audit binding",
                ..
            })
        ));
        let reassignment_event = AuditEvent::new(
            Some(original.session_id),
            Some(original.id),
            AuditKind::TranscriptSpeakerReassigned,
            original.capture_end_ns + 1_000,
            original.wall_clock_end + Duration::seconds(1),
            &reassigned,
            Some(original_event.hash.clone()),
        )
        .unwrap();
        store
            .append_transcript_speaker_reassignment_with_audit(&reassignment_event, &reassigned)
            .unwrap();
        assert!(store.verify().unwrap());
    }

    #[test]
    fn rejects_unknown_cross_session_and_merged_reassignment_targets_in_the_store() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let original = transcript_fixture(session_id);
        let original_event = transcript_event(&original, None);
        store
            .append_transcript_revision_with_audit(&original_event, &original)
            .unwrap();

        let active = SpeakerClusterRecord::new(session_id, 1).unwrap();
        let active_initial = SpeakerClusterLabelRevision::initial_generated(&active).unwrap();
        let active_event =
            speaker_creation(&active, &active_initial, Some(original_event.hash.clone()));
        store
            .append_speaker_cluster_with_audit(&active_event, &active, &active_initial)
            .unwrap();

        let merged = SpeakerClusterRecord::new(session_id, 2).unwrap();
        let merged_initial = SpeakerClusterLabelRevision::initial_generated(&merged).unwrap();
        let merged_event =
            speaker_creation(&merged, &merged_initial, Some(active_event.hash.clone()));
        store
            .append_speaker_cluster_with_audit(&merged_event, &merged, &merged_initial)
            .unwrap();

        let foreign = SpeakerClusterRecord::new(Uuid::new_v4(), 1).unwrap();
        let foreign_initial = SpeakerClusterLabelRevision::initial_generated(&foreign).unwrap();
        let foreign_event =
            speaker_creation(&foreign, &foreign_initial, Some(merged_event.hash.clone()));
        store
            .append_speaker_cluster_with_audit(&foreign_event, &foreign, &foreign_initial)
            .unwrap();

        let alias =
            SpeakerClusterAliasRevision::aliased_to(merged.id.clone(), active.id.clone()).unwrap();
        let alias_event = speaker_alias_event(&merged, &alias, Some(foreign_event.hash.clone()));
        store
            .append_speaker_cluster_alias_revision_with_audit(&alias_event, &alias)
            .unwrap();

        let unknown = SpeakerClusterRecord::new(session_id, 3).unwrap();
        for target_cluster_id in [unknown.id, foreign.id.clone(), merged.id.clone()] {
            let reassigned = TranscriptRevision::revision_of(
                &original,
                original.timing(),
                Some(target_cluster_id),
                original.text.clone(),
                true,
                TranscriptSource::UserEdited,
                original.model.clone(),
                original.confidence,
            )
            .unwrap();
            let event = speaker_reassignment_event(&reassigned, Some(alias_event.hash.clone()));
            assert!(matches!(
                store.append_transcript_speaker_reassignment_with_audit(&event, &reassigned),
                Err(AuditStoreError::InvalidTranscriptMetadata {
                    field: "speaker reassignment target",
                    ..
                })
            ));
        }

        let reassigned = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            Some(active.id.clone()),
            original.text.clone(),
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            original.confidence,
        )
        .unwrap();
        let event = speaker_reassignment_event(&reassigned, Some(alias_event.hash));
        store
            .append_transcript_speaker_reassignment_with_audit(&event, &reassigned)
            .unwrap();
        assert!(store.verify().unwrap());
    }

    #[test]
    fn rejects_a_forged_unknown_dedicated_reassignment_after_reopen() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-speaker-reassignment-{}.sqlite3",
            Uuid::new_v4()
        ));
        let session_id = Uuid::new_v4();
        let original = transcript_fixture(session_id);
        let original_event = transcript_event(&original, None);
        let active = SpeakerClusterRecord::new(session_id, 1).unwrap();
        let initial = SpeakerClusterLabelRevision::initial_generated(&active).unwrap();
        let active_event = speaker_creation(&active, &initial, Some(original_event.hash.clone()));
        let unknown = SpeakerClusterRecord::new(session_id, 2).unwrap();
        let forged = TranscriptRevision::revision_of(
            &original,
            original.timing(),
            Some(unknown.id),
            original.text.clone(),
            true,
            TranscriptSource::UserEdited,
            original.model.clone(),
            original.confidence,
        )
        .unwrap();
        let forged_event = speaker_reassignment_event(&forged, Some(active_event.hash.clone()));

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_transcript_revision_with_audit(&original_event, &original)
                .unwrap();
            store
                .append_speaker_cluster_with_audit(&active_event, &active, &initial)
                .unwrap();

            // Simulate an on-disk attacker that forges a fully chained event
            // and matching row, bypassing the checked Store entrypoint.
            insert_audit_event(&store.connection, &forged_event).unwrap();
            insert_transcript_revision(&store.connection, &forged).unwrap();
            assert!(!store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert!(!reopened.verify().unwrap());
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }
}
