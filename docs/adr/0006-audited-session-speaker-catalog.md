# ADR-0006: Keep Speaker Corrections Session-Scoped and Append-Only

## Status

Accepted

## Context

WordCovenant records final transcript revisions locally, but the existing
`speaker_cluster_id` field has no catalog, ownership boundary, correction
history, or durable display-label semantics. The product must let a user group
and correct speakers without claiming an inferred human identity, retaining
voice embeddings, or rewriting transcript evidence.

## Decision

M3.0 introduces a session-scoped anonymous speaker catalog. Each cluster uses
an opaque locally generated ID and has append-only label and alias revisions.
The generated label is an anonymous ordinal such as `Speaker 1`; a user label
is presentation metadata and is never an identity assertion. Cluster aliases
represent a reversible merge without rewriting historical transcript rows.

Changing a transcript's speaker assignment creates a new final
`TranscriptRevision` with source `UserEdited`. It preserves the original text,
capture range, wall-clock range, and model provenance. The action is bound to a
dedicated audit event and cannot overwrite a prior revision.

No M3.0 record stores PCM, embeddings, voice profiles, a personal identity,
confidence, or cross-session linkage. The WebView receives only compact cluster
projections and final transcript projections. This introduces no network
client, model download path, or egress permission.

## Consequences

### Positive

- Users can correct speaker grouping while retaining an auditable history.
- M2.2 native capture and inference outcomes remain isolated from speaker UI
  actions and retain their existing timing and privacy guarantees.
- Later local diarization can assign opaque cluster IDs without changing the
  correction and evidence model.

### Negative

- M3.0 does not automatically derive clusters from speech. Users create and
  correct anonymous groups manually until a separately benchmarked local
  embedding engine exists.
- Merge and split operations need optimistic-concurrency checks and atomic
  SQLite bundles because multiple immutable records can change together.

## Alternatives Considered

**Overwrite `speaker_cluster_id` in place:** Rejected because it destroys
correction history and invalidates the transcript audit contract.

**Use display strings such as `speaker-1` as persistent identity:** Rejected
because IDs can collide, cannot support rename history, and tempt callers to
treat a label as a verified person.

**Persist voice embeddings in M3.0:** Rejected because it expands biometric
data handling before a local model, calibration, consent, retention, and
deletion design exist.
