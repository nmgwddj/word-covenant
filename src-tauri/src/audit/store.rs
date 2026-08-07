use super::{AuditEvent, AuditTrail};
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
            ",
        )?;
        Ok(Self { connection })
    }

    pub fn append(&self, event: &AuditEvent) -> Result<(), AuditStoreError> {
        self.connection.execute(
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
    use crate::audit::AuditKind;

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
}
