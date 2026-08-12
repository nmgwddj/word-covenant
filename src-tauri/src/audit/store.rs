use super::{AuditEvent, AuditKind, AuditTrail};
use crate::audio::{CaptureGap, CapturePoint, SpeechDetectionSettings};
use crate::diarization::{SpeakerEmbedding, SpeakerSampleQuality};
use crate::domain::session::SessionDeletedAuditPayload;
use crate::domain::{
    CaptureSegment, CaptureSession, SpeakerCluster, SpeakerClusterAliasRevision,
    SpeakerClusterCreatedAuditPayload, SpeakerClusterLabelRevision, SpeakerClusterRecord,
    SpeakerObservation, SpeakerObservationAuditPayload, SpeakerObservationDecision,
    SpeakerPrototype, SpeakerPrototypeAuditBinding, TranscriptModelProvenance, TranscriptRevision,
    TranscriptSource, VoiceProfile, VoiceProfileAuditBinding, VoiceProfileCreatedAuditPayload,
    VoiceProfileDeletedAuditPayload, VoiceProfileEnrollmentAuditPayload,
    VoiceProfileRevisionAuditPayload, VoiceProfileState,
};
use crate::inference::asr::logical_span_id_for_asr_utterance_digest;
use crate::inference::model_registry::{LocalModelKind, RegisteredModel};
use crate::inference::{
    AsrFinalIdempotencyKey, InferenceGap, InferenceGapReason, InferenceGapStage,
};
use chrono::{DateTime, Utc};
use rusqlite::{params, types::Value, Connection, OptionalExtension};
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
    InvalidVoiceProfileMetadata {
        field: &'static str,
        value: String,
    },
    InvalidSessionDeletionMetadata {
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
            Self::InvalidVoiceProfileMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid voice profile metadata for {field}: {value}"
                )
            }
            Self::InvalidSessionDeletionMetadata { field, value } => {
                write!(
                    formatter,
                    "invalid session deletion metadata for {field}: {value}"
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

const IMMUTABLE_DELETE_TRIGGERS_DROP_SQL: &str = "
    DROP TRIGGER inference_gaps_are_immutable_delete;
    DROP TRIGGER transcript_revisions_are_immutable_delete;
    DROP TRIGGER speaker_clusters_are_immutable_delete;
    DROP TRIGGER speaker_cluster_label_revisions_are_immutable_delete;
    DROP TRIGGER speaker_cluster_alias_revisions_are_immutable_delete;
    DROP TRIGGER asr_final_idempotency_is_immutable_delete;
    DROP TRIGGER speaker_observations_are_immutable_delete;
";

const IMMUTABLE_DELETE_TRIGGERS_CREATE_SQL: &str = "
    CREATE TRIGGER inference_gaps_are_immutable_delete
    BEFORE DELETE ON inference_gaps
    BEGIN
        SELECT RAISE(ABORT, 'inference gaps are immutable');
    END;
    CREATE TRIGGER transcript_revisions_are_immutable_delete
    BEFORE DELETE ON transcript_revisions
    BEGIN
        SELECT RAISE(ABORT, 'transcript revisions are immutable');
    END;
    CREATE TRIGGER speaker_clusters_are_immutable_delete
    BEFORE DELETE ON speaker_clusters
    BEGIN
        SELECT RAISE(ABORT, 'speaker clusters are immutable');
    END;
    CREATE TRIGGER speaker_cluster_label_revisions_are_immutable_delete
    BEFORE DELETE ON speaker_cluster_label_revisions
    BEGIN
        SELECT RAISE(ABORT, 'speaker label revisions are immutable');
    END;
    CREATE TRIGGER speaker_cluster_alias_revisions_are_immutable_delete
    BEFORE DELETE ON speaker_cluster_alias_revisions
    BEGIN
        SELECT RAISE(ABORT, 'speaker alias revisions are immutable');
    END;
    CREATE TRIGGER asr_final_idempotency_is_immutable_delete
    BEFORE DELETE ON asr_final_idempotency
    BEGIN
        SELECT RAISE(ABORT, 'ASR final idempotency records are immutable');
    END;
    CREATE TRIGGER speaker_observations_are_immutable_delete
    BEFORE DELETE ON speaker_observations
    BEGIN
        SELECT RAISE(ABORT, 'speaker observations are immutable');
    END;
";

const VOICE_PROFILE_DELETE_TRIGGERS_DROP_SQL: &str = "
    DROP TRIGGER speaker_profiles_are_immutable_delete;
    DROP TRIGGER speaker_profile_prototypes_are_immutable_delete;
";

const VOICE_PROFILE_DELETE_TRIGGERS_CREATE_SQL: &str = "
    CREATE TRIGGER speaker_profiles_are_immutable_delete
    BEFORE DELETE ON speaker_profiles
    BEGIN
        SELECT RAISE(ABORT, 'speaker profiles are immutable');
    END;
    CREATE TRIGGER speaker_profile_prototypes_are_immutable_delete
    BEFORE DELETE ON speaker_profile_prototypes
    BEGIN
        SELECT RAISE(ABORT, 'speaker profile prototypes are immutable');
    END;
";

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

            CREATE TABLE IF NOT EXISTS speaker_profiles (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                revision_id TEXT NOT NULL UNIQUE,
                parent_revision_id TEXT,
                revision INTEGER NOT NULL CHECK (revision > 0),
                display_name TEXT NOT NULL,
                state TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                confirmed_duration_ns TEXT NOT NULL,
                learning_started_at TEXT NOT NULL,
                origin_session_id TEXT,
                origin_cluster_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                audit_event_id TEXT NOT NULL UNIQUE,
                UNIQUE(profile_id, revision)
            );
            CREATE INDEX IF NOT EXISTS speaker_profiles_profile_sequence
                ON speaker_profiles(profile_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_profiles_are_immutable_update
            BEFORE UPDATE ON speaker_profiles
            BEGIN
                SELECT RAISE(ABORT, 'speaker profiles are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_profiles_are_immutable_delete
            BEFORE DELETE ON speaker_profiles
            BEGIN
                SELECT RAISE(ABORT, 'speaker profiles are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_profile_prototypes (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                profile_id TEXT NOT NULL,
                profile_revision_id TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                embedding BLOB NOT NULL,
                confirmed_duration_ns TEXT NOT NULL,
                confirmed_at TEXT NOT NULL,
                source_observation_id TEXT UNIQUE,
                audit_event_id TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS speaker_profile_prototypes_profile_sequence
                ON speaker_profile_prototypes(profile_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_profile_prototypes_are_immutable_update
            BEFORE UPDATE ON speaker_profile_prototypes
            BEGIN
                SELECT RAISE(ABORT, 'speaker profile prototypes are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_profile_prototypes_are_immutable_delete
            BEFORE DELETE ON speaker_profile_prototypes
            BEGIN
                SELECT RAISE(ABORT, 'speaker profile prototypes are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_observations (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                transcript_revision_id TEXT NOT NULL UNIQUE,
                profile_id TEXT,
                anonymous_cluster_id TEXT,
                label_snapshot TEXT,
                decision TEXT NOT NULL,
                similarity REAL,
                runner_up_similarity REAL,
                model_provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                embedding BLOB NOT NULL,
                voiced_duration_ns TEXT NOT NULL,
                voiced_ratio REAL NOT NULL,
                signal_quality REAL NOT NULL,
                overlap_probability REAL NOT NULL,
                observed_at TEXT NOT NULL,
                audit_event_id TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS speaker_observations_session_sequence
                ON speaker_observations(session_id, sequence);
            CREATE INDEX IF NOT EXISTS speaker_observations_profile_sequence
                ON speaker_observations(profile_id, sequence);
            CREATE TRIGGER IF NOT EXISTS speaker_observations_are_immutable_update
            BEFORE UPDATE ON speaker_observations
            BEGIN
                SELECT RAISE(ABORT, 'speaker observations are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_observations_are_immutable_delete
            BEFORE DELETE ON speaker_observations
            BEGIN
                SELECT RAISE(ABORT, 'speaker observations are immutable');
            END;

            CREATE TABLE IF NOT EXISTS speaker_profile_deletions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL UNIQUE,
                purged_audit_event_ids TEXT NOT NULL,
                purged_audit_event_ids_sha256 TEXT NOT NULL,
                purged_audit_event_count INTEGER NOT NULL CHECK (purged_audit_event_count > 0),
                audit_event_id TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS speaker_profile_deletions_are_immutable_update
            BEFORE UPDATE ON speaker_profile_deletions
            BEGIN
                SELECT RAISE(ABORT, 'speaker profile deletions are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS speaker_profile_deletions_are_immutable_delete
            BEFORE DELETE ON speaker_profile_deletions
            BEGIN
                SELECT RAISE(ABORT, 'speaker profile deletions are immutable');
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

            CREATE TABLE IF NOT EXISTS capture_preferences (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                mode TEXT NOT NULL DEFAULT 'adaptive'
                    CHECK (mode IN ('adaptive', 'manual')),
                rms_threshold_dbfs INTEGER NOT NULL
                    CHECK (rms_threshold_dbfs BETWEEN -42 AND 0)
            );
            ",
        )?;
        ensure_column(
            &connection,
            "speaker_profiles",
            "learning_started_at",
            "TEXT",
        )?;
        connection.execute(
            "UPDATE speaker_profiles SET learning_started_at = created_at WHERE learning_started_at IS NULL",
            [],
        )?;
        ensure_column(&connection, "speaker_profiles", "origin_session_id", "TEXT")?;
        ensure_column(&connection, "speaker_profiles", "origin_cluster_id", "TEXT")?;
        ensure_column(
            &connection,
            "speaker_profile_prototypes",
            "source_observation_id",
            "TEXT",
        )?;
        connection.execute_batch(
            "
            CREATE UNIQUE INDEX IF NOT EXISTS speaker_profiles_origin_cluster
                ON speaker_profiles(origin_session_id, origin_cluster_id)
                WHERE revision = 1 AND origin_session_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS speaker_profile_prototypes_source_observation
                ON speaker_profile_prototypes(source_observation_id)
                WHERE source_observation_id IS NOT NULL;
            ",
        )?;
        Ok(Self { connection })
    }

    /// Read the mutable local capture preference. A missing row or a row that
    /// cannot satisfy the current validation contract falls back to the safe
    /// product default instead of becoming an active runtime threshold.
    pub fn speech_detection_settings(&self) -> Result<SpeechDetectionSettings, AuditStoreError> {
        let persisted = self
            .connection
            .query_row(
                "SELECT mode, rms_threshold_dbfs FROM capture_preferences WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, Value>(0)?, row.get::<_, Value>(1)?)),
            )
            .optional()?;
        Ok(match persisted {
            Some((Value::Text(mode), Value::Integer(value))) => {
                SpeechDetectionSettings::from_persisted_values(&mode, value).unwrap_or_default()
            }
            Some(_) | None => SpeechDetectionSettings::default(),
        })
    }

    pub fn set_speech_detection_settings(
        &mut self,
        settings: SpeechDetectionSettings,
    ) -> Result<(), AuditStoreError> {
        settings
            .validate()
            .map_err(|_| AuditStoreError::InvalidCaptureMetadata {
                field: "speech detection settings",
                value: format!("{}:{}", settings.mode.as_str(), settings.rms_threshold_dbfs),
            })?;
        self.connection.execute(
            "
            INSERT INTO capture_preferences (singleton, mode, rms_threshold_dbfs)
            VALUES (1, ?1, ?2)
            ON CONFLICT(singleton) DO UPDATE SET
                mode = excluded.mode,
                rms_threshold_dbfs = excluded.rms_threshold_dbfs
            ",
            params![settings.mode.as_str(), settings.rms_threshold_dbfs],
        )?;
        Ok(())
    }

    pub fn append(&self, event: &AuditEvent) -> Result<(), AuditStoreError> {
        insert_audit_event(&self.connection, event)
    }

    /// Appends a minimal deletion tombstone and removes all content-bearing
    /// records owned by the session in the same SQLite transaction.
    pub fn delete_session_with_audit(
        &mut self,
        event: &AuditEvent,
        payload: &SessionDeletedAuditPayload,
    ) -> Result<(), AuditStoreError> {
        validate_session_deleted_audit_event(&self.connection, event, payload)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;

        // SQLite DDL is transactional. Dropping these guards only within this
        // connection lets the product's explicit erase operation remove
        // immutable content while rollback restores every guard on failure.
        transaction.execute_batch(IMMUTABLE_DELETE_TRIGGERS_DROP_SQL)?;
        let session_id = payload.session_id.to_string();
        transaction.execute(
            "DELETE FROM speaker_observations WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM transcript_revision_fts WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM asr_final_idempotency WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "
            DELETE FROM speaker_cluster_label_revisions
            WHERE speaker_cluster_id IN (
                SELECT id FROM speaker_clusters WHERE session_id = ?1
            )
            ",
            params![&session_id],
        )?;
        transaction.execute(
            "
            DELETE FROM speaker_cluster_alias_revisions
            WHERE speaker_cluster_id IN (
                SELECT id FROM speaker_clusters WHERE session_id = ?1
            )
            ",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM speaker_clusters WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM transcript_revisions WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM inference_gaps WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM capture_gaps WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute(
            "DELETE FROM capture_segments WHERE session_id = ?1",
            params![&session_id],
        )?;
        transaction.execute_batch(IMMUTABLE_DELETE_TRIGGERS_CREATE_SQL)?;
        transaction.commit()?;
        Ok(())
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
        let enrollment_clusters =
            speaker_clusters_with_enrollment_samples(&self.connection, session_id)?;
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
                    enrollment_clusters.contains(&stored.record.id),
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

    pub fn append_voice_profile_with_audit(
        &mut self,
        event: &AuditEvent,
        profile: &VoiceProfile,
    ) -> Result<(), AuditStoreError> {
        let was_deleted = self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM speaker_profile_deletions WHERE profile_id = ?1)",
            params![profile.id.to_string()],
            |row| row.get::<_, bool>(0),
        )?;
        if was_deleted {
            return Err(voice_profile_error(
                "profile creation",
                "a deleted voice profile ID cannot be reused",
            ));
        }
        validate_voice_profile_created_audit_event(event, profile)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_voice_profile(&transaction, profile, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn append_voice_profile_revision_with_audit(
        &mut self,
        event: &AuditEvent,
        profile: &VoiceProfile,
    ) -> Result<(), AuditStoreError> {
        let previous = latest_voice_profile(&self.connection, profile.id)?
            .ok_or_else(|| voice_profile_error("profile revision", "profile does not exist"))?;
        profile
            .validate_successor_of(&previous)
            .map_err(|value| voice_profile_error("profile revision", value))?;
        validate_voice_profile_revision_audit_event(event, profile)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_voice_profile(&transaction, profile, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn append_speaker_label_and_voice_profile_revision_with_audit(
        &mut self,
        label_event: &AuditEvent,
        label: &SpeakerClusterLabelRevision,
        profile_event: &AuditEvent,
        profile: &VoiceProfile,
    ) -> Result<(), AuditStoreError> {
        let cluster = validate_speaker_cluster_label_revision_for_write(&self.connection, label)?;
        validate_speaker_cluster_label_revision_audit_event(label_event, &cluster, label)?;
        let previous = latest_voice_profile(&self.connection, profile.id)?
            .ok_or_else(|| voice_profile_error("profile revision", "profile does not exist"))?;
        profile
            .validate_successor_of(&previous)
            .map_err(|value| voice_profile_error("profile revision", value))?;
        if profile.origin_session_id != Some(cluster.session_id)
            || profile.origin_cluster_id.as_deref() != Some(cluster.id.as_str())
            || profile.display_name != label.label
        {
            return Err(voice_profile_error(
                "profile label revision",
                "speaker cluster and voice profile names must stay aligned",
            ));
        }
        validate_voice_profile_revision_audit_event(profile_event, profile)?;

        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, label_event)?;
        insert_speaker_cluster_label_revision(&transaction, label, label_event.id)?;
        insert_audit_event(&transaction, profile_event)?;
        insert_voice_profile(&transaction, profile, profile_event.id)?;
        transaction.commit()?;
        Ok(())
    }

    /// Atomically appends one user-confirmed prototype and the resulting
    /// profile learning-state revision. Automatic matches must not call this.
    pub fn append_voice_profile_enrollment_with_audit(
        &mut self,
        event: &AuditEvent,
        profile: &VoiceProfile,
        prototype: &SpeakerPrototype,
    ) -> Result<(), AuditStoreError> {
        let previous = latest_voice_profile(&self.connection, profile.id)?
            .ok_or_else(|| voice_profile_error("profile enrollment", "profile does not exist"))?;
        profile
            .validate_successor_of(&previous)
            .map_err(|value| voice_profile_error("profile enrollment", value))?;
        prototype
            .validate_for_profile(profile)
            .map_err(|value| voice_profile_error("prototype", value))?;
        if prototype.source_observation_id.is_some() {
            validate_prototype_source_observation(&self.connection, profile, prototype)?;
        }
        let expected_duration = previous
            .confirmed_duration_ns
            .saturating_add(prototype.confirmed_duration_ns)
            .min(crate::domain::voice_profile::MAX_CONFIRMED_DURATION_NS);
        if profile.confirmed_duration_ns != expected_duration {
            return Err(voice_profile_error(
                "profile enrollment duration",
                format!(
                    "expected {expected_duration}, got {}",
                    profile.confirmed_duration_ns
                ),
            ));
        }
        let prototype_count: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM speaker_profile_prototypes WHERE profile_id = ?1",
            params![profile.id.to_string()],
            |row| row.get(0),
        )?;
        if prototype_count >= crate::domain::voice_profile::MAX_PROTOTYPES_PER_PROFILE as i64 {
            return Err(voice_profile_error(
                "prototype count",
                prototype_count.to_string(),
            ));
        }
        validate_voice_profile_enrollment_audit_event(event, profile, prototype)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_voice_profile(&transaction, profile, event.id)?;
        insert_speaker_prototype(&transaction, prototype, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    /// Atomically confirms an anonymous cluster as one persistent local voice
    /// profile. Every prototype in this bundle comes from an explicitly
    /// confirmed observation in that exact session cluster.
    pub fn append_voice_profile_cluster_enrollment_with_audit(
        &mut self,
        label_event: &AuditEvent,
        label: &SpeakerClusterLabelRevision,
        profile_events: &[AuditEvent],
        profiles: &[VoiceProfile],
        prototypes: &[SpeakerPrototype],
    ) -> Result<(), AuditStoreError> {
        if profiles.is_empty()
            || profiles.len() != profile_events.len()
            || profiles.len() != prototypes.len() + 1
        {
            return Err(voice_profile_error(
                "cluster enrollment bundle",
                "profile event and prototype counts do not form a revision chain",
            ));
        }
        let cluster = validate_speaker_cluster_label_revision_for_write(&self.connection, label)?;
        validate_speaker_cluster_label_revision_audit_event(label_event, &cluster, label)?;
        let initial = &profiles[0];
        if initial.origin_session_id != Some(cluster.session_id)
            || initial.origin_cluster_id.as_deref() != Some(cluster.id.as_str())
            || initial.display_name != label.label
        {
            return Err(voice_profile_error(
                "cluster enrollment origin",
                "voice profile does not bind the confirmed speaker cluster",
            ));
        }
        validate_voice_profile_created_audit_event(&profile_events[0], initial)?;
        for ((previous, current), (event, prototype)) in profiles
            .windows(2)
            .map(|pair| (&pair[0], &pair[1]))
            .zip(profile_events[1..].iter().zip(prototypes))
        {
            current
                .validate_successor_of(previous)
                .map_err(|value| voice_profile_error("cluster enrollment profile chain", value))?;
            prototype
                .validate_for_profile(current)
                .map_err(|value| voice_profile_error("cluster enrollment prototype", value))?;
            let source =
                validate_prototype_source_observation(&self.connection, current, prototype)?;
            if source.session_id != cluster.session_id
                || source.anonymous_cluster_id.as_deref() != Some(cluster.id.as_str())
                || source.decision != SpeakerObservationDecision::AnonymousCluster
            {
                return Err(voice_profile_error(
                    "cluster enrollment observation",
                    "confirmed observation does not belong to the anonymous cluster",
                ));
            }
            validate_voice_profile_enrollment_audit_event(event, current, prototype)?;
        }

        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, label_event)?;
        insert_speaker_cluster_label_revision(&transaction, label, label_event.id)?;
        insert_audit_event(&transaction, &profile_events[0])?;
        insert_voice_profile(&transaction, initial, profile_events[0].id)?;
        for ((profile, prototype), event) in profiles[1..]
            .iter()
            .zip(prototypes)
            .zip(&profile_events[1..])
        {
            insert_audit_event(&transaction, event)?;
            insert_voice_profile(&transaction, profile, event.id)?;
            insert_speaker_prototype(&transaction, prototype, event.id)?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn append_speaker_observation_with_audit(
        &mut self,
        event: &AuditEvent,
        observation: &SpeakerObservation,
    ) -> Result<(), AuditStoreError> {
        validate_speaker_observation_audit_event(event, observation)?;
        validate_speaker_observation_for_write(&self.connection, observation)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        insert_speaker_observation(&transaction, observation, event.id)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_voice_profiles(&self) -> Result<Vec<VoiceProfile>, AuditStoreError> {
        list_current_voice_profiles(&self.connection)
    }

    pub fn list_speaker_prototypes(
        &self,
        profile_id: Uuid,
    ) -> Result<Vec<SpeakerPrototype>, AuditStoreError> {
        let Some(profile) = latest_voice_profile(&self.connection, profile_id)? else {
            return Ok(Vec::new());
        };
        list_speaker_prototypes(&self.connection, Some(profile_id)).map(|records| {
            records
                .into_iter()
                .map(|record| record.value)
                .filter(|prototype| {
                    prototype.embedding.model() == &profile.model
                        && prototype.confirmed_at >= profile.learning_started_at
                })
                .collect()
        })
    }

    pub fn list_speaker_observations(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<SpeakerObservation>, AuditStoreError> {
        list_speaker_observations(&self.connection, Some(session_id))
            .map(|records| records.into_iter().map(|record| record.value).collect())
    }

    pub fn list_all_speaker_observations(
        &self,
    ) -> Result<Vec<SpeakerObservation>, AuditStoreError> {
        list_speaker_observations(&self.connection, None)
            .map(|records| records.into_iter().map(|record| record.value).collect())
    }

    pub fn voice_profile_deletion_payload(
        &self,
        profile_id: Uuid,
    ) -> Result<VoiceProfileDeletedAuditPayload, AuditStoreError> {
        if latest_voice_profile(&self.connection, profile_id)?.is_none() {
            return Err(voice_profile_error(
                "profile deletion",
                "profile does not exist",
            ));
        }
        let audit_event_ids = voice_profile_purged_audit_event_ids(&self.connection, profile_id)?;
        VoiceProfileDeletedAuditPayload::new(profile_id, &audit_event_ids)
            .map_err(|value| voice_profile_error("profile deletion payload", value))
    }

    /// Physically erases biometric profile records and vectors. Historical
    /// observation label snapshots remain immutable, while this minimal
    /// tombstone prevents future recognition under the deleted profile ID.
    pub fn delete_voice_profile_with_audit(
        &mut self,
        event: &AuditEvent,
        payload: &VoiceProfileDeletedAuditPayload,
    ) -> Result<(), AuditStoreError> {
        if latest_voice_profile(&self.connection, payload.profile_id)?.is_none() {
            return Err(voice_profile_error(
                "profile deletion",
                "profile does not exist",
            ));
        }
        let purged_audit_event_ids =
            voice_profile_purged_audit_event_ids(&self.connection, payload.profile_id)?;
        let expected_payload =
            VoiceProfileDeletedAuditPayload::new(payload.profile_id, &purged_audit_event_ids)
                .map_err(|value| voice_profile_error("profile deletion payload", value))?;
        if payload != &expected_payload {
            return Err(voice_profile_error(
                "profile deletion payload",
                "purged audit event digest does not match current profile records",
            ));
        }
        validate_voice_profile_deleted_audit_event(event, payload)?;
        let transaction = self.connection.transaction()?;
        insert_audit_event(&transaction, event)?;
        transaction.execute_batch(VOICE_PROFILE_DELETE_TRIGGERS_DROP_SQL)?;
        transaction.execute(
            "DELETE FROM speaker_profile_prototypes WHERE profile_id = ?1",
            params![payload.profile_id.to_string()],
        )?;
        transaction.execute(
            "DELETE FROM speaker_profiles WHERE profile_id = ?1",
            params![payload.profile_id.to_string()],
        )?;
        transaction.execute_batch(VOICE_PROFILE_DELETE_TRIGGERS_CREATE_SQL)?;
        transaction.execute(
            "
            INSERT INTO speaker_profile_deletions (
                profile_id, purged_audit_event_ids, purged_audit_event_ids_sha256,
                purged_audit_event_count, audit_event_id
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ",
            params![
                payload.profile_id.to_string(),
                purged_audit_event_ids
                    .iter()
                    .map(Uuid::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"),
                &payload.purged_audit_event_ids_sha256,
                i64::try_from(payload.purged_audit_event_count).map_err(|_| {
                    voice_profile_error(
                        "profile deletion event count",
                        payload.purged_audit_event_count.to_string(),
                    )
                })?,
                event.id.to_string()
            ],
        )?;
        transaction.commit()?;
        Ok(())
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

        let Some(deleted_session_ids) = verified_deleted_session_ids(&events)? else {
            return Ok(false);
        };
        let retained_events = events
            .iter()
            .filter(|event| {
                event
                    .run_id
                    .is_none_or(|session_id| !deleted_session_ids.contains(&session_id))
            })
            .cloned()
            .collect::<Vec<_>>();

        // The hash chain proves event ordering; these checks additionally
        // prove that every durable M2 record still matches exactly one event
        // after reopening the SQLite database.
        let transcript_events = retained_events
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

        let model_events = retained_events
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

        let inference_events = retained_events
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
        if !verify_speaker_catalog(&retained_events, &catalog) {
            return Ok(false);
        }

        if !verify_voice_profile_storage(&self.connection, &retained_events)? {
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

    /// Commits one native final transcript and its speaker decision as a
    /// single unit. A newly inferred session speaker is created before the
    /// transcript references it; the observation is appended last because it
    /// is causally bound to the immutable transcript revision.
    #[allow(clippy::too_many_arguments)]
    pub fn append_asr_final_with_speaker_audit(
        &mut self,
        transcript_event: &AuditEvent,
        revision: &TranscriptRevision,
        idempotency: &AsrFinalIdempotencyBinding,
        cluster_creation: Option<(
            &AuditEvent,
            &SpeakerClusterRecord,
            &SpeakerClusterLabelRevision,
        )>,
        profile_label: Option<(&AuditEvent, &SpeakerClusterLabelRevision)>,
        observation: Option<(&AuditEvent, &SpeakerObservation)>,
    ) -> Result<(), AuditStoreError> {
        validate_asr_final_audit_event(transcript_event, revision, idempotency)?;
        if observation.is_none() && (cluster_creation.is_some() || profile_label.is_some()) {
            return Err(voice_profile_error(
                "native speaker transaction",
                "a speaker cluster cannot be created without an observation",
            ));
        }

        if let Some((event, cluster, initial_label)) = cluster_creation {
            validate_speaker_cluster_created_audit_event(event, cluster, initial_label)?;
            if revision.speaker_cluster_id.as_deref() != Some(cluster.id.as_str()) {
                return Err(voice_profile_error(
                    "native speaker cluster",
                    "transcript does not reference its newly created speaker cluster",
                ));
            }
            if let Some((label_event, label)) = profile_label {
                label
                    .validate_successor_of(initial_label)
                    .map_err(|value| speaker_error("speaker label parent", value))?;
                validate_speaker_cluster_label_revision_audit_event(label_event, cluster, label)?;
            }
        } else if profile_label.is_some() {
            return Err(voice_profile_error(
                "native speaker label",
                "a profile label requires a newly created session cluster",
            ));
        }

        let transaction = self.connection.transaction()?;
        if let Some((event, cluster, initial_label)) = cluster_creation {
            insert_audit_event(&transaction, event)?;
            insert_speaker_cluster(&transaction, cluster, event.id)?;
            insert_speaker_cluster_label_revision(&transaction, initial_label, event.id)?;
        }
        if let Some((event, label)) = profile_label {
            insert_audit_event(&transaction, event)?;
            insert_speaker_cluster_label_revision(&transaction, label, event.id)?;
        }
        insert_audit_event(&transaction, transcript_event)?;
        insert_transcript_revision(&transaction, revision)?;
        insert_asr_final_idempotency(&transaction, idempotency, revision.id)?;
        if let Some((event, observation)) = observation {
            validate_speaker_observation_for_write(&transaction, observation)?;
            validate_speaker_observation_audit_event(event, observation)?;
            insert_audit_event(&transaction, event)?;
            insert_speaker_observation(&transaction, observation, event.id)?;
        }
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

struct VoiceProfileRow {
    profile_id: String,
    revision_id: String,
    parent_revision_id: Option<String>,
    revision: i64,
    display_name: String,
    state: String,
    model_provider: String,
    model_id: String,
    model_version: String,
    model_sha256: String,
    confirmed_duration_ns: String,
    learning_started_at: String,
    origin_session_id: Option<String>,
    origin_cluster_id: Option<String>,
    created_at: String,
    updated_at: String,
    audit_event_id: String,
}

struct SpeakerPrototypeRow {
    id: String,
    profile_id: String,
    profile_revision_id: String,
    model_provider: String,
    model_id: String,
    model_version: String,
    model_sha256: String,
    dimensions: i64,
    embedding: Vec<u8>,
    confirmed_duration_ns: String,
    confirmed_at: String,
    source_observation_id: Option<String>,
    audit_event_id: String,
}

struct SpeakerObservationRow {
    id: String,
    session_id: String,
    transcript_revision_id: String,
    profile_id: Option<String>,
    anonymous_cluster_id: Option<String>,
    label_snapshot: Option<String>,
    decision: String,
    similarity: Option<f64>,
    runner_up_similarity: Option<f64>,
    model_provider: String,
    model_id: String,
    model_version: String,
    model_sha256: String,
    dimensions: i64,
    embedding: Vec<u8>,
    voiced_duration_ns: String,
    voiced_ratio: f64,
    signal_quality: f64,
    overlap_probability: f64,
    observed_at: String,
    audit_event_id: String,
}

#[derive(Clone)]
struct AuditedRecord<T> {
    value: T,
    audit_event_id: Uuid,
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

fn voice_profile_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<VoiceProfileRow> {
    Ok(VoiceProfileRow {
        profile_id: row.get(0)?,
        revision_id: row.get(1)?,
        parent_revision_id: row.get(2)?,
        revision: row.get(3)?,
        display_name: row.get(4)?,
        state: row.get(5)?,
        model_provider: row.get(6)?,
        model_id: row.get(7)?,
        model_version: row.get(8)?,
        model_sha256: row.get(9)?,
        confirmed_duration_ns: row.get(10)?,
        learning_started_at: row.get(11)?,
        origin_session_id: row.get(12)?,
        origin_cluster_id: row.get(13)?,
        created_at: row.get(14)?,
        updated_at: row.get(15)?,
        audit_event_id: row.get(16)?,
    })
}

fn speaker_prototype_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpeakerPrototypeRow> {
    Ok(SpeakerPrototypeRow {
        id: row.get(0)?,
        profile_id: row.get(1)?,
        profile_revision_id: row.get(2)?,
        model_provider: row.get(3)?,
        model_id: row.get(4)?,
        model_version: row.get(5)?,
        model_sha256: row.get(6)?,
        dimensions: row.get(7)?,
        embedding: row.get(8)?,
        confirmed_duration_ns: row.get(9)?,
        confirmed_at: row.get(10)?,
        source_observation_id: row.get(11)?,
        audit_event_id: row.get(12)?,
    })
}

fn speaker_observation_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpeakerObservationRow> {
    Ok(SpeakerObservationRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        transcript_revision_id: row.get(2)?,
        profile_id: row.get(3)?,
        anonymous_cluster_id: row.get(4)?,
        label_snapshot: row.get(5)?,
        decision: row.get(6)?,
        similarity: row.get(7)?,
        runner_up_similarity: row.get(8)?,
        model_provider: row.get(9)?,
        model_id: row.get(10)?,
        model_version: row.get(11)?,
        model_sha256: row.get(12)?,
        dimensions: row.get(13)?,
        embedding: row.get(14)?,
        voiced_duration_ns: row.get(15)?,
        voiced_ratio: row.get(16)?,
        signal_quality: row.get(17)?,
        overlap_probability: row.get(18)?,
        observed_at: row.get(19)?,
        audit_event_id: row.get(20)?,
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

fn parse_speaker_model_provenance(
    provider: String,
    model_id: String,
    version: String,
    sha256: String,
) -> Result<crate::inference::ModelProvenance, AuditStoreError> {
    crate::inference::ModelProvenance::new(provider, model_id, version, sha256)
        .map_err(|value| voice_profile_error("model provenance", value))
}

fn parse_voice_profile(
    row: VoiceProfileRow,
) -> Result<AuditedRecord<VoiceProfile>, AuditStoreError> {
    let profile = VoiceProfile {
        id: parse_uuid(&row.profile_id)?,
        revision_id: parse_uuid(&row.revision_id)?,
        parent_revision_id: parse_optional_uuid(row.parent_revision_id)?,
        revision: row
            .revision
            .try_into()
            .map_err(|_| voice_profile_error("profile revision", row.revision.to_string()))?,
        display_name: row.display_name,
        state: serde_json::from_str(&row.state)
            .map_err(|_| voice_profile_error("profile state", row.state))?,
        model: parse_speaker_model_provenance(
            row.model_provider,
            row.model_id,
            row.model_version,
            row.model_sha256,
        )?,
        confirmed_duration_ns: row.confirmed_duration_ns.parse().map_err(|_| {
            voice_profile_error("profile confirmed duration", row.confirmed_duration_ns)
        })?,
        learning_started_at: parse_timestamp(&row.learning_started_at)?,
        origin_session_id: parse_optional_uuid(row.origin_session_id)?,
        origin_cluster_id: row.origin_cluster_id,
        created_at: parse_timestamp(&row.created_at)?,
        updated_at: parse_timestamp(&row.updated_at)?,
    };
    profile
        .validate()
        .map_err(|value| voice_profile_error("profile", value))?;
    Ok(AuditedRecord {
        value: profile,
        audit_event_id: parse_uuid(&row.audit_event_id)?,
    })
}

fn parse_speaker_embedding(
    model: crate::inference::ModelProvenance,
    dimensions: i64,
    bytes: Vec<u8>,
) -> Result<SpeakerEmbedding, AuditStoreError> {
    let dimensions: usize = dimensions
        .try_into()
        .map_err(|_| voice_profile_error("embedding dimensions", dimensions.to_string()))?;
    if bytes.len() != dimensions.saturating_mul(std::mem::size_of::<f32>()) {
        return Err(voice_profile_error(
            "embedding bytes",
            format!("dimensions={dimensions}, bytes={}", bytes.len()),
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    SpeakerEmbedding::from_normalized(model, values)
        .map_err(|value| voice_profile_error("embedding", value))
}

fn parse_speaker_prototype(
    row: SpeakerPrototypeRow,
) -> Result<AuditedRecord<SpeakerPrototype>, AuditStoreError> {
    let model = parse_speaker_model_provenance(
        row.model_provider,
        row.model_id,
        row.model_version,
        row.model_sha256,
    )?;
    let prototype = SpeakerPrototype {
        id: parse_uuid(&row.id)?,
        profile_id: parse_uuid(&row.profile_id)?,
        profile_revision_id: parse_uuid(&row.profile_revision_id)?,
        embedding: parse_speaker_embedding(model, row.dimensions, row.embedding)?,
        confirmed_duration_ns: row.confirmed_duration_ns.parse().map_err(|_| {
            voice_profile_error("prototype confirmed duration", row.confirmed_duration_ns)
        })?,
        confirmed_at: parse_timestamp(&row.confirmed_at)?,
        source_observation_id: parse_optional_uuid(row.source_observation_id)?,
    };
    Ok(AuditedRecord {
        value: prototype,
        audit_event_id: parse_uuid(&row.audit_event_id)?,
    })
}

fn parse_speaker_observation(
    row: SpeakerObservationRow,
) -> Result<AuditedRecord<SpeakerObservation>, AuditStoreError> {
    let model = parse_speaker_model_provenance(
        row.model_provider,
        row.model_id,
        row.model_version,
        row.model_sha256,
    )?;
    let quality = SpeakerSampleQuality::new(
        row.voiced_duration_ns.parse().map_err(|_| {
            voice_profile_error("observation voiced duration", row.voiced_duration_ns)
        })?,
        finite_f32("observation voiced ratio", row.voiced_ratio)?,
        finite_f32("observation signal quality", row.signal_quality)?,
        finite_f32("observation overlap probability", row.overlap_probability)?,
    )
    .map_err(|value| voice_profile_error("observation quality", value))?;
    let observation = SpeakerObservation::new(
        parse_uuid(&row.id)?,
        parse_uuid(&row.session_id)?,
        parse_uuid(&row.transcript_revision_id)?,
        row.profile_id.as_deref().map(parse_uuid).transpose()?,
        row.anonymous_cluster_id,
        row.label_snapshot,
        serde_json::from_str::<SpeakerObservationDecision>(&row.decision)
            .map_err(|_| voice_profile_error("observation decision", row.decision))?,
        row.similarity
            .map(|value| finite_f32("observation similarity", value))
            .transpose()?,
        row.runner_up_similarity
            .map(|value| finite_f32("observation runner-up similarity", value))
            .transpose()?,
        parse_speaker_embedding(model, row.dimensions, row.embedding)?,
        quality,
        parse_timestamp(&row.observed_at)?,
    )
    .map_err(|value| voice_profile_error("observation", value))?;
    Ok(AuditedRecord {
        value: observation,
        audit_event_id: parse_uuid(&row.audit_event_id)?,
    })
}

fn finite_f32(field: &'static str, value: f64) -> Result<f32, AuditStoreError> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(voice_profile_error(field, value.to_string()));
    }
    Ok(value as f32)
}

fn list_voice_profile_revisions(
    connection: &Connection,
) -> Result<Vec<AuditedRecord<VoiceProfile>>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT profile_id, revision_id, parent_revision_id, revision, display_name, state,
               model_provider, model_id, model_version, model_sha256,
               confirmed_duration_ns, learning_started_at, origin_session_id, origin_cluster_id,
               created_at, updated_at, audit_event_id
        FROM speaker_profiles
        ORDER BY sequence ASC
        ",
    )?;
    let rows = statement.query_map([], voice_profile_row)?;
    rows.map(|row| parse_voice_profile(row?)).collect()
}

fn list_current_voice_profiles(
    connection: &Connection,
) -> Result<Vec<VoiceProfile>, AuditStoreError> {
    let revisions = list_voice_profile_revisions(connection)?;
    let mut current = BTreeMap::<Uuid, VoiceProfile>::new();
    for revision in revisions {
        if let Some(previous) = current.get(&revision.value.id) {
            revision
                .value
                .validate_successor_of(previous)
                .map_err(|value| voice_profile_error("profile revision chain", value))?;
        } else if revision.value.revision != 1 {
            return Err(voice_profile_error(
                "profile revision chain",
                "profile does not start at revision one",
            ));
        }
        current.insert(revision.value.id, revision.value);
    }
    Ok(current.into_values().collect())
}

fn latest_voice_profile(
    connection: &Connection,
    profile_id: Uuid,
) -> Result<Option<VoiceProfile>, AuditStoreError> {
    Ok(list_current_voice_profiles(connection)?
        .into_iter()
        .find(|profile| profile.id == profile_id))
}

fn list_speaker_prototypes(
    connection: &Connection,
    profile_id: Option<Uuid>,
) -> Result<Vec<AuditedRecord<SpeakerPrototype>>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT id, profile_id, profile_revision_id, model_provider, model_id,
               model_version, model_sha256, dimensions, embedding,
               confirmed_duration_ns, confirmed_at, source_observation_id, audit_event_id
        FROM speaker_profile_prototypes
        WHERE (?1 IS NULL OR profile_id = ?1)
        ORDER BY sequence ASC
        ",
    )?;
    let stored_profile_id = profile_id.map(|value| value.to_string());
    let rows = statement.query_map(params![stored_profile_id], speaker_prototype_row)?;
    rows.map(|row| parse_speaker_prototype(row?)).collect()
}

fn list_speaker_observations(
    connection: &Connection,
    session_id: Option<Uuid>,
) -> Result<Vec<AuditedRecord<SpeakerObservation>>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT id, session_id, transcript_revision_id, profile_id, anonymous_cluster_id,
               label_snapshot, decision, similarity, runner_up_similarity,
               model_provider, model_id, model_version, model_sha256, dimensions, embedding,
               voiced_duration_ns, voiced_ratio, signal_quality, overlap_probability,
               observed_at, audit_event_id
        FROM speaker_observations
        WHERE (?1 IS NULL OR session_id = ?1)
        ORDER BY sequence ASC
        ",
    )?;
    let stored_session_id = session_id.map(|value| value.to_string());
    let rows = statement.query_map(params![stored_session_id], speaker_observation_row)?;
    rows.map(|row| parse_speaker_observation(row?)).collect()
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

fn speaker_clusters_with_enrollment_samples(
    connection: &Connection,
    session_id: Uuid,
) -> Result<BTreeSet<String>, AuditStoreError> {
    let decision = serde_json::to_string(&SpeakerObservationDecision::AnonymousCluster)
        .expect("speaker observation decision serializes");
    let mut statement = connection.prepare(
        "
        SELECT DISTINCT anonymous_cluster_id
        FROM speaker_observations
        WHERE session_id = ?1
          AND anonymous_cluster_id IS NOT NULL
          AND decision = ?2
        ",
    )?;
    let rows = statement.query_map(params![session_id.to_string(), decision], |row| row.get(0))?;
    rows.collect::<Result<BTreeSet<String>, _>>()
        .map_err(AuditStoreError::from)
}

fn validate_session_deleted_audit_event(
    connection: &Connection,
    event: &AuditEvent,
    payload: &SessionDeletedAuditPayload,
) -> Result<(), AuditStoreError> {
    if event.kind != AuditKind::SessionDeleted {
        return Err(AuditStoreError::InvalidSessionDeletionMetadata {
            field: "audit event kind",
            value: format!("{:?}", event.kind),
        });
    }
    if event.run_id != Some(payload.session_id) || event.causation_id.is_some() {
        return Err(AuditStoreError::InvalidSessionDeletionMetadata {
            field: "audit event linkage",
            value: format!("run={:?}, causation={:?}", event.run_id, event.causation_id),
        });
    }
    if !event.matches_payload(payload).map_err(|error| {
        AuditStoreError::InvalidSessionDeletionMetadata {
            field: "audit event payload",
            value: error.to_string(),
        }
    })? {
        return Err(AuditStoreError::InvalidSessionDeletionMetadata {
            field: "audit event payload",
            value: "digest does not match".to_owned(),
        });
    }

    let session_id = payload.session_id.to_string();
    let session_started_kind =
        serde_json::to_string(&AuditKind::SessionStarted).expect("audit kind is serializable");
    let session_deleted_kind =
        serde_json::to_string(&AuditKind::SessionDeleted).expect("audit kind is serializable");
    let (started_count, deleted_count) = connection.query_row(
        "
        SELECT
            SUM(CASE WHEN kind = ?2 THEN 1 ELSE 0 END),
            SUM(CASE WHEN kind = ?3 THEN 1 ELSE 0 END)
        FROM audit_events
        WHERE run_id = ?1
        ",
        params![session_id, session_started_kind, session_deleted_kind],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    if started_count != 1 || deleted_count != 0 {
        return Err(AuditStoreError::InvalidSessionDeletionMetadata {
            field: "session lifecycle",
            value: format!("started={started_count}, deleted={deleted_count}"),
        });
    }
    Ok(())
}

fn validate_voice_profile_created_audit_event(
    event: &AuditEvent,
    profile: &VoiceProfile,
) -> Result<(), AuditStoreError> {
    if profile.revision != 1 || profile.parent_revision_id.is_some() {
        return Err(voice_profile_error(
            "profile creation",
            "initial profile must be revision one",
        ));
    }
    let payload = VoiceProfileCreatedAuditPayload {
        profile: VoiceProfileAuditBinding::from_profile(profile)
            .map_err(|value| voice_profile_error("profile creation", value))?,
    };
    validate_voice_profile_event(
        event,
        AuditKind::VoiceProfileCreated,
        None,
        None,
        profile.updated_at,
        &payload,
    )
}

fn validate_voice_profile_revision_audit_event(
    event: &AuditEvent,
    profile: &VoiceProfile,
) -> Result<(), AuditStoreError> {
    let payload = VoiceProfileRevisionAuditPayload {
        profile: VoiceProfileAuditBinding::from_profile(profile)
            .map_err(|value| voice_profile_error("profile revision", value))?,
    };
    validate_voice_profile_event(
        event,
        AuditKind::VoiceProfileRevisionRecorded,
        None,
        profile.parent_revision_id,
        profile.updated_at,
        &payload,
    )
}

fn validate_voice_profile_enrollment_audit_event(
    event: &AuditEvent,
    profile: &VoiceProfile,
    prototype: &SpeakerPrototype,
) -> Result<(), AuditStoreError> {
    let payload = VoiceProfileEnrollmentAuditPayload {
        profile: VoiceProfileAuditBinding::from_profile(profile)
            .map_err(|value| voice_profile_error("profile enrollment", value))?,
        prototype: SpeakerPrototypeAuditBinding::from_prototype(prototype),
    };
    validate_voice_profile_event(
        event,
        AuditKind::VoiceProfileEnrollmentRecorded,
        None,
        profile.parent_revision_id,
        profile.updated_at,
        &payload,
    )
}

fn validate_speaker_observation_audit_event(
    event: &AuditEvent,
    observation: &SpeakerObservation,
) -> Result<(), AuditStoreError> {
    let payload = SpeakerObservationAuditPayload {
        observation: crate::domain::SpeakerObservationAuditBinding::from_observation(observation)
            .map_err(|value| voice_profile_error("speaker observation", value))?,
    };
    validate_voice_profile_event(
        event,
        AuditKind::SpeakerObservationRecorded,
        Some(observation.session_id),
        Some(observation.transcript_revision_id),
        observation.observed_at,
        &payload,
    )
}

fn validate_speaker_observation_for_write(
    connection: &Connection,
    observation: &SpeakerObservation,
) -> Result<(), AuditStoreError> {
    let transcript_session = connection
        .query_row(
            "SELECT session_id FROM transcript_revisions WHERE id = ?1",
            params![observation.transcript_revision_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if transcript_session.as_deref() != Some(observation.session_id.to_string().as_str()) {
        return Err(voice_profile_error(
            "observation transcript",
            "final transcript revision does not belong to the observation session",
        ));
    }
    if let Some(cluster_id) = observation.anonymous_cluster_id.as_deref() {
        let cluster_session = connection
            .query_row(
                "SELECT session_id FROM speaker_clusters WHERE id = ?1",
                params![cluster_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if cluster_session.as_deref() != Some(observation.session_id.to_string().as_str()) {
            return Err(voice_profile_error(
                "observation anonymous cluster",
                "speaker cluster does not belong to the observation session",
            ));
        }
    }
    if observation.decision == SpeakerObservationDecision::MatchedProfile {
        let profile_id = observation.profile_id.ok_or_else(|| {
            voice_profile_error("observation profile", "matched observation has no profile")
        })?;
        let profile = latest_voice_profile(connection, profile_id)?.ok_or_else(|| {
            voice_profile_error("observation profile", "matched profile does not exist")
        })?;
        if profile.state != VoiceProfileState::Ready {
            return Err(voice_profile_error(
                "observation profile",
                "only ready profiles may be matched automatically",
            ));
        }
        if observation.embedding.model() != &profile.model {
            return Err(voice_profile_error(
                "observation profile",
                "matched observation uses an incompatible model space",
            ));
        }
    }
    Ok(())
}

fn validate_prototype_source_observation(
    connection: &Connection,
    profile: &VoiceProfile,
    prototype: &SpeakerPrototype,
) -> Result<SpeakerObservation, AuditStoreError> {
    let source_observation_id = prototype.source_observation_id.ok_or_else(|| {
        voice_profile_error(
            "prototype source observation",
            "user-confirmed prototypes must identify their source observation",
        )
    })?;
    let observation = list_speaker_observations(connection, None)?
        .into_iter()
        .find(|record| record.value.id == source_observation_id)
        .map(|record| record.value)
        .ok_or_else(|| {
            voice_profile_error(
                "prototype source observation",
                "source observation does not exist",
            )
        })?;
    if observation.embedding != prototype.embedding
        || observation.quality.voiced_duration_ns() != prototype.confirmed_duration_ns
        || observation.embedding.model() != &profile.model
        || observation.observed_at > prototype.confirmed_at
    {
        return Err(voice_profile_error(
            "prototype source observation",
            "source observation does not match the confirmed prototype",
        ));
    }
    Ok(observation)
}

fn validate_voice_profile_deleted_audit_event(
    event: &AuditEvent,
    payload: &VoiceProfileDeletedAuditPayload,
) -> Result<(), AuditStoreError> {
    validate_voice_profile_event(
        event,
        AuditKind::VoiceProfileDeleted,
        None,
        None,
        event.wall_clock,
        payload,
    )
}

fn validate_voice_profile_event<T: Serialize>(
    event: &AuditEvent,
    kind: AuditKind,
    run_id: Option<Uuid>,
    causation_id: Option<Uuid>,
    wall_clock: DateTime<Utc>,
    payload: &T,
) -> Result<(), AuditStoreError> {
    if event.kind != kind
        || event.run_id != run_id
        || event.causation_id != causation_id
        || event.wall_clock != wall_clock
    {
        return Err(voice_profile_error(
            "audit linkage",
            format!(
                "kind={:?}, run={:?}, causation={:?}, wall_clock={}",
                event.kind, event.run_id, event.causation_id, event.wall_clock
            ),
        ));
    }
    if !event
        .matches_payload(payload)
        .map_err(|value| voice_profile_error("audit payload", value.to_string()))?
    {
        return Err(voice_profile_error(
            "audit payload",
            "digest does not match",
        ));
    }
    Ok(())
}

fn voice_profile_purged_audit_event_ids(
    connection: &Connection,
    profile_id: Uuid,
) -> Result<Vec<Uuid>, AuditStoreError> {
    let profile_id = profile_id.to_string();
    let mut statement = connection.prepare(
        "
        SELECT audit_event_id FROM speaker_profiles WHERE profile_id = ?1
        UNION
        SELECT audit_event_id FROM speaker_profile_prototypes WHERE profile_id = ?1
        ORDER BY audit_event_id ASC
        ",
    )?;
    let rows = statement.query_map(params![profile_id], |row| row.get::<_, String>(0))?;
    rows.map(|row| parse_uuid(&row?)).collect()
}

fn verified_deleted_session_ids(
    events: &[AuditEvent],
) -> Result<Option<BTreeSet<Uuid>>, AuditStoreError> {
    let mut started = BTreeSet::new();
    let mut deleted = BTreeSet::new();
    for event in events {
        match event.kind {
            AuditKind::SessionStarted => {
                let Some(session_id) = event.run_id else {
                    continue;
                };
                started.insert(session_id);
            }
            AuditKind::SessionDeleted => {
                let Some(session_id) = event.run_id else {
                    return Ok(None);
                };
                let payload = SessionDeletedAuditPayload { session_id };
                let payload_matches = event.matches_payload(&payload).map_err(|error| {
                    AuditStoreError::InvalidSessionDeletionMetadata {
                        field: "retained audit payload",
                        value: error.to_string(),
                    }
                })?;
                if event.causation_id.is_some()
                    || !payload_matches
                    || !started.contains(&session_id)
                    || !deleted.insert(session_id)
                {
                    return Ok(None);
                }
            }
            _ => {}
        }
    }
    Ok(Some(deleted))
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

fn verify_voice_profile_storage(
    connection: &Connection,
    events: &[AuditEvent],
) -> Result<bool, AuditStoreError> {
    let events_by_id = events
        .iter()
        .map(|event| (event.id, event))
        .collect::<BTreeMap<_, _>>();
    let revisions = match list_voice_profile_revisions(connection) {
        Ok(revisions) => revisions,
        Err(_) => return Ok(false),
    };
    let profiles_by_revision = revisions
        .iter()
        .map(|profile| (profile.value.revision_id, &profile.value))
        .collect::<BTreeMap<_, _>>();
    let prototypes = match list_speaker_prototypes(connection, None) {
        Ok(prototypes) => prototypes,
        Err(_) => return Ok(false),
    };
    let observations = match list_speaker_observations(connection, None) {
        Ok(observations) => observations,
        Err(_) => return Ok(false),
    };
    let deleted = load_voice_profile_deletions(connection)?;
    let purged_event_ids = deleted
        .iter()
        .flat_map(|deletion| deletion.purged_audit_event_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut bound_event_ids = BTreeSet::new();

    for revision in &revisions {
        let Some(event) = events_by_id.get(&revision.audit_event_id) else {
            return Ok(false);
        };
        let valid = if revision.value.revision == 1 {
            validate_voice_profile_created_audit_event(event, &revision.value).is_ok()
        } else if event.kind == AuditKind::VoiceProfileEnrollmentRecorded {
            let Some(prototype) = prototypes.iter().find(|prototype| {
                prototype.value.profile_revision_id == revision.value.revision_id
            }) else {
                return Ok(false);
            };
            validate_voice_profile_enrollment_audit_event(event, &revision.value, &prototype.value)
                .is_ok()
        } else {
            validate_voice_profile_revision_audit_event(event, &revision.value).is_ok()
        };
        if !valid || !bound_event_ids.insert(revision.audit_event_id) {
            return Ok(false);
        }
    }
    for prototype in &prototypes {
        let Some(profile) = profiles_by_revision.get(&prototype.value.profile_revision_id) else {
            return Ok(false);
        };
        let Some(event) = events_by_id.get(&prototype.audit_event_id) else {
            return Ok(false);
        };
        let source_is_valid = prototype.value.source_observation_id.is_none()
            || validate_prototype_source_observation(connection, profile, &prototype.value).is_ok();
        if prototype.value.validate_for_profile(profile).is_err()
            || !source_is_valid
            || event.kind != AuditKind::VoiceProfileEnrollmentRecorded
            || validate_voice_profile_enrollment_audit_event(event, profile, &prototype.value)
                .is_err()
        {
            return Ok(false);
        }
    }
    for observation in &observations {
        let Some(event) = events_by_id.get(&observation.audit_event_id) else {
            return Ok(false);
        };
        if !bound_event_ids.insert(observation.audit_event_id)
            || validate_speaker_observation_audit_event(event, &observation.value).is_err()
        {
            return Ok(false);
        }
        let transcript_exists = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM transcript_revisions WHERE id = ?1 AND session_id = ?2)",
            params![
                observation.value.transcript_revision_id.to_string(),
                observation.value.session_id.to_string()
            ],
            |row| row.get::<_, bool>(0),
        )?;
        if !transcript_exists {
            return Ok(false);
        }
    }
    for deletion in &deleted {
        let Some(event) = events_by_id.get(&deletion.audit_event_id) else {
            return Ok(false);
        };
        if !bound_event_ids.insert(deletion.audit_event_id)
            || validate_voice_profile_deleted_audit_event(event, &deletion.payload).is_err()
            || deletion
                .purged_audit_event_ids
                .iter()
                .any(|id| !events_by_id.contains_key(id))
        {
            return Ok(false);
        }
    }

    let speaker_event_ids = events
        .iter()
        .filter(|event| {
            matches!(
                event.kind,
                AuditKind::VoiceProfileCreated
                    | AuditKind::VoiceProfileRevisionRecorded
                    | AuditKind::VoiceProfileEnrollmentRecorded
                    | AuditKind::VoiceProfileDeleted
                    | AuditKind::SpeakerObservationRecorded
            )
        })
        .map(|event| event.id)
        .collect::<BTreeSet<_>>();
    bound_event_ids.extend(purged_event_ids);
    Ok(bound_event_ids == speaker_event_ids)
}

struct VoiceProfileDeletionRecord {
    payload: VoiceProfileDeletedAuditPayload,
    purged_audit_event_ids: Vec<Uuid>,
    audit_event_id: Uuid,
}

fn load_voice_profile_deletions(
    connection: &Connection,
) -> Result<Vec<VoiceProfileDeletionRecord>, AuditStoreError> {
    let mut statement = connection.prepare(
        "
        SELECT profile_id, purged_audit_event_ids, purged_audit_event_ids_sha256,
               purged_audit_event_count, audit_event_id
        FROM speaker_profile_deletions
        ORDER BY sequence ASC
        ",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    rows.map(|row| {
        let (profile_id, event_ids, sha256, count, audit_event_id) = row?;
        let purged_audit_event_ids = event_ids
            .split('\n')
            .map(parse_uuid)
            .collect::<Result<Vec<_>, _>>()?;
        let count: usize = count
            .try_into()
            .map_err(|_| voice_profile_error("profile deletion event count", count.to_string()))?;
        let payload =
            VoiceProfileDeletedAuditPayload::new(parse_uuid(&profile_id)?, &purged_audit_event_ids)
                .map_err(|value| voice_profile_error("profile deletion payload", value))?;
        if payload.purged_audit_event_ids_sha256 != sha256
            || payload.purged_audit_event_count != count
        {
            return Err(voice_profile_error(
                "profile deletion binding",
                "persisted digest or count does not match event IDs",
            ));
        }
        Ok(VoiceProfileDeletionRecord {
            payload,
            purged_audit_event_ids,
            audit_event_id: parse_uuid(&audit_event_id)?,
        })
    })
    .collect()
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

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<(), AuditStoreError> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|value| value == column) {
        connection.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        ))?;
    }
    Ok(())
}

fn insert_voice_profile(
    connection: &Connection,
    profile: &VoiceProfile,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    profile
        .validate()
        .map_err(|value| voice_profile_error("profile", value))?;
    connection.execute(
        "
        INSERT INTO speaker_profiles (
            profile_id, revision_id, parent_revision_id, revision, display_name, state,
            model_provider, model_id, model_version, model_sha256,
            confirmed_duration_ns, learning_started_at, origin_session_id, origin_cluster_id,
            created_at, updated_at, audit_event_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
        )
        ",
        params![
            profile.id.to_string(),
            profile.revision_id.to_string(),
            profile.parent_revision_id.map(|value| value.to_string()),
            i64::from(profile.revision),
            &profile.display_name,
            serde_json::to_string(&profile.state).expect("voice profile state serializes"),
            profile.model.provider(),
            profile.model.model_id(),
            profile.model.model_version(),
            profile.model.artifact_sha256(),
            profile.confirmed_duration_ns.to_string(),
            profile.learning_started_at.to_rfc3339(),
            profile.origin_session_id.map(|value| value.to_string()),
            profile.origin_cluster_id.as_deref(),
            profile.created_at.to_rfc3339(),
            profile.updated_at.to_rfc3339(),
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_speaker_prototype(
    connection: &Connection,
    prototype: &SpeakerPrototype,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    connection.execute(
        "
        INSERT INTO speaker_profile_prototypes (
            id, profile_id, profile_revision_id, model_provider, model_id,
            model_version, model_sha256, dimensions, embedding,
            confirmed_duration_ns, confirmed_at, source_observation_id, audit_event_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
        ",
        params![
            prototype.id.to_string(),
            prototype.profile_id.to_string(),
            prototype.profile_revision_id.to_string(),
            prototype.embedding.model().provider(),
            prototype.embedding.model().model_id(),
            prototype.embedding.model().model_version(),
            prototype.embedding.model().artifact_sha256(),
            i64::try_from(prototype.embedding.dimensions()).map_err(|_| {
                voice_profile_error(
                    "prototype dimensions",
                    prototype.embedding.dimensions().to_string(),
                )
            })?,
            crate::domain::voice_profile::embedding_bytes(&prototype.embedding),
            prototype.confirmed_duration_ns.to_string(),
            prototype.confirmed_at.to_rfc3339(),
            prototype
                .source_observation_id
                .map(|value| value.to_string()),
            audit_event_id.to_string(),
        ],
    )?;
    Ok(())
}

fn insert_speaker_observation(
    connection: &Connection,
    observation: &SpeakerObservation,
    audit_event_id: Uuid,
) -> Result<(), AuditStoreError> {
    observation
        .validate()
        .map_err(|value| voice_profile_error("observation", value))?;
    connection.execute(
        "
        INSERT INTO speaker_observations (
            id, session_id, transcript_revision_id, profile_id, anonymous_cluster_id,
            label_snapshot, decision, similarity, runner_up_similarity,
            model_provider, model_id, model_version, model_sha256, dimensions, embedding,
            voiced_duration_ns, voiced_ratio, signal_quality, overlap_probability,
            observed_at, audit_event_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
            ?16, ?17, ?18, ?19, ?20, ?21
        )
        ",
        params![
            observation.id.to_string(),
            observation.session_id.to_string(),
            observation.transcript_revision_id.to_string(),
            observation.profile_id.map(|value| value.to_string()),
            &observation.anonymous_cluster_id,
            &observation.label_snapshot,
            serde_json::to_string(&observation.decision)
                .expect("speaker observation decision serializes"),
            observation.similarity.map(f64::from),
            observation.runner_up_similarity.map(f64::from),
            observation.embedding.model().provider(),
            observation.embedding.model().model_id(),
            observation.embedding.model().model_version(),
            observation.embedding.model().artifact_sha256(),
            i64::try_from(observation.embedding.dimensions()).map_err(|_| {
                voice_profile_error(
                    "observation dimensions",
                    observation.embedding.dimensions().to_string(),
                )
            })?,
            crate::domain::voice_profile::embedding_bytes(&observation.embedding),
            observation.quality.voiced_duration_ns().to_string(),
            f64::from(observation.quality.voiced_ratio()),
            f64::from(observation.quality.signal_quality()),
            f64::from(observation.quality.overlap_probability()),
            observation.observed_at.to_rfc3339(),
            audit_event_id.to_string(),
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

fn voice_profile_error(field: &'static str, value: impl Into<String>) -> AuditStoreError {
    AuditStoreError::InvalidVoiceProfileMetadata {
        field,
        value: value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{CaptureGapReason, SpeechDetectionSettings};
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
    fn speech_detection_settings_default_and_persist_across_reopen() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-speech-detection-settings-{}.sqlite3",
            Uuid::new_v4()
        ));

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            assert_eq!(
                store.speech_detection_settings().unwrap(),
                SpeechDetectionSettings::default()
            );
            let configured = SpeechDetectionSettings::new(-24).unwrap();
            store.set_speech_detection_settings(configured).unwrap();
            assert_eq!(store.speech_detection_settings().unwrap(), configured);
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(
            reopened.speech_detection_settings().unwrap(),
            SpeechDetectionSettings::new(-24).unwrap()
        );
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn speech_detection_settings_reject_invalid_values_without_replacing_the_saved_value() {
        let mut store = AuditStore::open_in_memory().unwrap();
        let configured = SpeechDetectionSettings::new(-24).unwrap();
        store.set_speech_detection_settings(configured).unwrap();

        assert!(store
            .set_speech_detection_settings(SpeechDetectionSettings {
                mode: crate::audio::SpeechDetectionMode::Manual,
                rms_threshold_dbfs: -43,
            })
            .is_err());
        assert_eq!(store.speech_detection_settings().unwrap(), configured);
    }

    #[test]
    fn speech_detection_settings_fall_back_to_default_for_an_invalid_persisted_row() {
        let store = AuditStore::open_in_memory().unwrap();
        store
            .connection
            .execute_batch("PRAGMA ignore_check_constraints = ON")
            .unwrap();
        store
            .connection
            .execute(
                "INSERT INTO capture_preferences (singleton, rms_threshold_dbfs) VALUES (1, ?1)",
                params![-43],
            )
            .unwrap();

        assert_eq!(
            store.speech_detection_settings().unwrap(),
            SpeechDetectionSettings::default()
        );
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

    fn speaker_model(version: &str) -> crate::inference::ModelProvenance {
        crate::inference::ModelProvenance::new(
            "fixture",
            "speaker-embedding",
            version,
            "d".repeat(64),
        )
        .unwrap()
    }

    fn speaker_embedding() -> SpeakerEmbedding {
        SpeakerEmbedding::new(speaker_model("v1"), vec![0.9, 0.1, 0.0]).unwrap()
    }

    fn profile_created_event(profile: &VoiceProfile, previous_hash: Option<String>) -> AuditEvent {
        let payload = VoiceProfileCreatedAuditPayload {
            profile: VoiceProfileAuditBinding::from_profile(profile).unwrap(),
        };
        AuditEvent::new(
            None,
            None,
            AuditKind::VoiceProfileCreated,
            1,
            profile.updated_at,
            &payload,
            previous_hash,
        )
        .unwrap()
    }

    fn profile_enrollment_event(
        profile: &VoiceProfile,
        prototype: &SpeakerPrototype,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        let payload = VoiceProfileEnrollmentAuditPayload {
            profile: VoiceProfileAuditBinding::from_profile(profile).unwrap(),
            prototype: SpeakerPrototypeAuditBinding::from_prototype(prototype),
        };
        AuditEvent::new(
            None,
            profile.parent_revision_id,
            AuditKind::VoiceProfileEnrollmentRecorded,
            2,
            profile.updated_at,
            &payload,
            previous_hash,
        )
        .unwrap()
    }

    fn observation_event(
        observation: &SpeakerObservation,
        previous_hash: Option<String>,
    ) -> AuditEvent {
        let payload = SpeakerObservationAuditPayload {
            observation: crate::domain::SpeakerObservationAuditBinding::from_observation(
                observation,
            )
            .unwrap(),
        };
        AuditEvent::new(
            Some(observation.session_id),
            Some(observation.transcript_revision_id),
            AuditKind::SpeakerObservationRecorded,
            3,
            observation.observed_at,
            &payload,
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
            assert!(!projected.can_enroll_voice_profile);
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

    #[test]
    fn persists_audited_voice_profiles_prototypes_and_observations_after_reopen() {
        let database = std::env::temp_dir().join(format!(
            "word-covenant-voice-profiles-{}.sqlite3",
            Uuid::new_v4()
        ));
        let now = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(40);
        let profile = VoiceProfile::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "Alice",
            speaker_model("v1"),
            now,
        )
        .unwrap();
        let created = profile_created_event(&profile, None);
        let learned = VoiceProfile::confirm_with_id(
            &profile,
            Uuid::new_v4(),
            4_000_000_000,
            now + Duration::seconds(1),
        )
        .unwrap();
        let prototype = SpeakerPrototype::new_with_id(
            Uuid::new_v4(),
            &learned,
            speaker_embedding(),
            4_000_000_000,
            learned.updated_at,
        )
        .unwrap();
        let enrolled = profile_enrollment_event(&learned, &prototype, Some(created.hash.clone()));
        let transcript = transcript_fixture(Uuid::new_v4());
        let transcript_event = transcript_event(&transcript, Some(enrolled.hash.clone()));
        let observation = SpeakerObservation::new(
            Uuid::new_v4(),
            transcript.session_id,
            transcript.id,
            Some(profile.id),
            None,
            Some("Alice".to_owned()),
            SpeakerObservationDecision::MatchedProfile,
            Some(0.93),
            Some(0.61),
            speaker_embedding(),
            SpeakerSampleQuality::new(1_200_000_000, 0.9, 0.8, 0.0).unwrap(),
            transcript.wall_clock_end,
        )
        .unwrap();
        let observed = observation_event(&observation, Some(transcript_event.hash.clone()));

        {
            let mut store = AuditStore::open_path(&database).unwrap();
            store
                .append_voice_profile_with_audit(&created, &profile)
                .unwrap();
            store
                .append_voice_profile_enrollment_with_audit(&enrolled, &learned, &prototype)
                .unwrap();
            store
                .append_transcript_revision_with_audit(&transcript_event, &transcript)
                .unwrap();
            store
                .append_speaker_observation_with_audit(&observed, &observation)
                .unwrap();
            assert_eq!(store.list_voice_profiles().unwrap(), vec![learned.clone()]);
            assert_eq!(
                store.list_speaker_prototypes(profile.id).unwrap(),
                vec![prototype.clone()]
            );
            assert_eq!(
                store
                    .list_speaker_observations(transcript.session_id)
                    .unwrap(),
                vec![observation.clone()]
            );
            assert!(store.verify().unwrap());
        }

        let reopened = AuditStore::open_path(&database).unwrap();
        assert_eq!(reopened.list_voice_profiles().unwrap(), vec![learned]);
        assert_eq!(
            reopened.list_speaker_prototypes(profile.id).unwrap(),
            vec![prototype]
        );
        assert_eq!(
            reopened
                .list_speaker_observations(transcript.session_id)
                .unwrap(),
            vec![observation]
        );
        assert!(reopened.verify().unwrap());
        drop(reopened);
        std::fs::remove_file(database).unwrap();
    }

    #[test]
    fn detects_tampered_voice_vectors_and_physically_deletes_profiles() {
        let now = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(50);
        let profile = VoiceProfile::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "Alice",
            speaker_model("v1"),
            now,
        )
        .unwrap();
        let created = profile_created_event(&profile, None);
        let learned = VoiceProfile::confirm_with_id(
            &profile,
            Uuid::new_v4(),
            4_000_000_000,
            now + Duration::seconds(1),
        )
        .unwrap();
        let prototype = SpeakerPrototype::new_with_id(
            Uuid::new_v4(),
            &learned,
            speaker_embedding(),
            4_000_000_000,
            learned.updated_at,
        )
        .unwrap();
        let enrolled = profile_enrollment_event(&learned, &prototype, Some(created.hash.clone()));
        let mut store = AuditStore::open_in_memory().unwrap();
        store
            .append_voice_profile_with_audit(&created, &profile)
            .unwrap();
        store
            .append_voice_profile_enrollment_with_audit(&enrolled, &learned, &prototype)
            .unwrap();
        assert!(store.verify().unwrap());

        let original_embedding =
            crate::domain::voice_profile::embedding_bytes(&speaker_embedding());
        store
            .connection
            .execute(
                "UPDATE speaker_profile_prototypes SET embedding = ?1 WHERE id = ?2",
                params![
                    vec![0_u8; original_embedding.len()],
                    prototype.id.to_string()
                ],
            )
            .unwrap_err();
        store
            .connection
            .execute_batch("DROP TRIGGER speaker_profile_prototypes_are_immutable_update;")
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE speaker_profile_prototypes SET embedding = ?1 WHERE id = ?2",
                params![
                    vec![0_u8; original_embedding.len()],
                    prototype.id.to_string()
                ],
            )
            .unwrap();
        assert!(!store.verify().unwrap());

        store
            .connection
            .execute(
                "UPDATE speaker_profile_prototypes SET embedding = ?1 WHERE id = ?2",
                params![original_embedding, prototype.id.to_string()],
            )
            .unwrap();
        assert!(store.verify().unwrap());
        let payload = store.voice_profile_deletion_payload(profile.id).unwrap();
        let deleted = AuditEvent::new(
            None,
            None,
            AuditKind::VoiceProfileDeleted,
            4,
            now + Duration::seconds(2),
            &payload,
            Some(enrolled.hash.clone()),
        )
        .unwrap();
        store
            .delete_voice_profile_with_audit(&deleted, &payload)
            .unwrap();
        assert!(store.list_voice_profiles().unwrap().is_empty());
        assert!(store
            .list_speaker_prototypes(profile.id)
            .unwrap()
            .is_empty());
        assert!(store.verify().unwrap());
    }

    #[test]
    fn session_deletion_removes_observations_but_preserves_voice_profiles() {
        let now = DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(60);
        let profile = VoiceProfile::new_with_id(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "Alice",
            speaker_model("v1"),
            now,
        )
        .unwrap();
        let created = profile_created_event(&profile, None);
        let transcript = transcript_fixture(Uuid::new_v4());
        let session_started = AuditEvent::new(
            Some(transcript.session_id),
            None,
            AuditKind::SessionStarted,
            1,
            transcript.wall_clock_start - Duration::seconds(1),
            &serde_json::json!({ "sessionId": transcript.session_id }),
            Some(created.hash.clone()),
        )
        .unwrap();
        let transcript_event = transcript_event(&transcript, Some(session_started.hash.clone()));
        let observation = SpeakerObservation::new(
            Uuid::new_v4(),
            transcript.session_id,
            transcript.id,
            None,
            None,
            None,
            SpeakerObservationDecision::Unknown,
            Some(0.93),
            None,
            speaker_embedding(),
            SpeakerSampleQuality::new(1_200_000_000, 0.9, 0.8, 0.0).unwrap(),
            transcript.wall_clock_end,
        )
        .unwrap();
        let observed = observation_event(&observation, Some(transcript_event.hash.clone()));
        let deletion_payload = SessionDeletedAuditPayload {
            session_id: transcript.session_id,
        };
        let deleted = AuditEvent::new(
            Some(transcript.session_id),
            None,
            AuditKind::SessionDeleted,
            transcript.capture_end_ns + 1,
            transcript.wall_clock_end + Duration::seconds(1),
            &deletion_payload,
            Some(observed.hash.clone()),
        )
        .unwrap();

        let mut store = AuditStore::open_in_memory().unwrap();
        store
            .append_voice_profile_with_audit(&created, &profile)
            .unwrap();
        store.append(&session_started).unwrap();
        store
            .append_transcript_revision_with_audit(&transcript_event, &transcript)
            .unwrap();
        store
            .append_speaker_observation_with_audit(&observed, &observation)
            .unwrap();
        store
            .delete_session_with_audit(&deleted, &deletion_payload)
            .unwrap();

        assert!(store
            .list_speaker_observations(transcript.session_id)
            .unwrap()
            .is_empty());
        assert_eq!(store.list_voice_profiles().unwrap(), vec![profile]);
        assert!(store.verify().unwrap());
    }
}
