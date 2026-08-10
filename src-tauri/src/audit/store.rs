use super::{AuditEvent, AuditKind, AuditTrail};
use crate::audio::{CaptureGap, CapturePoint};
use crate::domain::{
    CaptureSegment, CaptureSession, TranscriptModelProvenance, TranscriptRevision, TranscriptSource,
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
            .filter(|event| event.kind == AuditKind::TranscriptRevisionRecorded)
            .collect::<Vec<_>>();
        let revisions = self.list_all_transcript_revisions()?;
        let Some(asr_bindings) = self.verified_asr_final_idempotency_bindings()? else {
            return Ok(false);
        };
        if transcript_events.len() != revisions.len()
            || !revisions.iter().all(|revision| {
                let matches_event = |event: &&AuditEvent| match asr_bindings.get(&revision.id) {
                    Some(binding) => {
                        validate_asr_final_audit_event(event, revision, binding).is_ok()
                    }
                    None => validate_transcript_audit_event(event, revision).is_ok(),
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
}
