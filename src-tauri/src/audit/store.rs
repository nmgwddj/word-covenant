use super::{AuditEvent, AuditTrail};
use crate::audio::{CaptureGap, CapturePoint};
use crate::domain::CaptureSegment;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::Path;
use uuid::Uuid;

#[derive(Debug)]
pub enum AuditStoreError {
    Database(rusqlite::Error),
    InvalidUuid(String),
    InvalidTimestamp(String),
    InvalidKind(String),
    InvalidCaptureGapReason(String),
    InvalidCaptureMetadata { field: &'static str, value: String },
    InvalidCaptureGapRange,
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
            ",
        )?;
        Ok(Self { connection })
    }

    pub fn append(&self, event: &AuditEvent) -> Result<(), AuditStoreError> {
        insert_audit_event(&self.connection, event)
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
        let trail = AuditTrail::from_events(self.list()?);
        Ok(trail.verify())
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
}

fn insert_audit_event(connection: &Connection, event: &AuditEvent) -> Result<(), AuditStoreError> {
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

fn parse_capture_monotonic_ns(value: &str) -> Result<u64, AuditStoreError> {
    value
        .parse()
        .map_err(|_| AuditStoreError::InvalidCaptureMetadata {
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
    use chrono::Duration;

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
}
