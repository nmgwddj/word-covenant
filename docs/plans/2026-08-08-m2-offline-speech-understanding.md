# M2 Offline Speech Understanding

## Scope

This batch establishes the local, verifiable boundary required before a native
speech runtime is introduced. It deliberately does not bundle model weights,
download models, create an HTTP client, or route PCM through Tauri IPC.

The delivered vertical slice must prove that a deterministic local ASR fixture
can create revisioned final transcript records with capture timestamps and an
auditable model provenance record.

## Model import contract

`import_local_model` accepts a user-selected regular file and explicit model
metadata: model kind, format, version, model-card reference, license text or
identifier, and a license acknowledgement. The Rust process calculates
SHA-256 while importing, stores the file below the application-local model
directory, and records only its managed relative path. It never follows a
remote URL, invokes a downloader, or enables egress.

An imported record contains an immutable ID, model kind and format, byte size,
SHA-256, model version, model-card/license metadata, acknowledgement time, and
managed local path. The audit event contains the record's hashable provenance,
not model bytes or an arbitrary source path. A future removal workflow must be
audited separately; this batch has no implicit replacement or upgrade action.

## Inference boundary

The capture callback remains limited to its existing bounded PCM ingress. A
future Rust-only worker will consume that ingress, convert its input to 16 kHz
mono, apply VAD pre-roll/hangover, and pass fixed-size windows to a selected
ASR provider. No callback, model provider, or transcript store may perform
network I/O.

This batch defines provider traits and deterministic fixtures only. The VAD
fixture validates segmentation rules at 16 kHz. The ASR fixture emits explicit
partial or final results. `whisper.cpp` with Metal is a later adapter behind
the same trait, after the import and benchmark paths are stable.

## Transcript and audit contract

Each transcript lineage has a stable span ID; a newer revision is appended,
never used to overwrite the model output. Final records include capture start
and end, wall-clock anchor, source, model ID/version, and optional confidence.
Partial output is transient display state and is excluded from Agent context.

Final revisions are persisted with an FTS projection and their corresponding
audit event in one SQLite transaction. On restart, the timeline is hydrated
from the local store. Agent-facing context will later select only final
revisions. The first fixture test must demonstrate that a final ASR emission
is visible in the timeline, survives reopening the database, and leaves a
valid audit chain.

## Acceptance checks

1. Import rejects absent, empty, non-regular, or unacknowledged model files.
2. SHA-256 and byte count match the copied local artifact; no network code is
   introduced.
3. Fixture VAD/ASR output becomes a final, revisioned local transcript record.
4. Transcript storage rejects duplicate or regressive revisions and preserves
   prior revisions.
5. Focused Rust tests, full Rust tests, frontend tests/type-check, formatting,
   and a release compile check pass before a native runtime is added.
