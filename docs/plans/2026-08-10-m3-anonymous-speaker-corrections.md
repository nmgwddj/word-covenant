# M3 Anonymous Speaker Corrections Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an offline, auditable workflow for anonymous speaker groups and final transcript corrections without asserting a person's identity.

**Architecture:** A session-scoped SQLite catalog owns immutable cluster, label, and alias revisions. A speaker reassignment appends a `UserEdited` `TranscriptRevision`, retaining text, capture time, wall-clock time, and model provenance. `AppState` validates optimistic concurrency, commits audit-bound SQLite records, and projects compact data to Vue.

**Tech Stack:** Rust, Tauri 2, rusqlite, existing SHA-256 audit chain, Vue 3, TypeScript, Pinia, Vitest. No embedding model, model download, network client, or PCM IPC.

---

## Scope and non-goals

- M3.0 is manual anonymous correction only. It excludes voiceprints, identity matching, automatic diarization, confidence, overlap detection, embeddings, and cross-session clustering.
- A generated `Speaker N` label and a user-entered label are presentation metadata, never an identity claim.
- PCM, embeddings, transcript text in logs, and outbound requests remain out of scope. Legacy/mock `speaker-1` values are not migrated or treated as catalog records.
- All corrections target a final current-head transcript revision. A stale or invalid request produces no SQLite write and no audit event.

## Task 1: Define append-only anonymous speaker records

**Files:**
- Create: `src-tauri/src/domain/speaker.rs`
- Modify: `src-tauri/src/domain/mod.rs`
- Modify: `src-tauri/src/audit/hash_chain.rs`
- Test: `src-tauri/src/domain/speaker.rs`

1. Write failing tests for opaque generated IDs, positive anonymous ordinals, generated labels, valid user labels, empty/control-character/overlong label rejection, strict label and alias revision increments, self-alias rejection, and alias clearing by append.
2. Run `cargo test --manifest-path src-tauri/Cargo.toml --offline domain::speaker --lib`; it should fail before the module exists.
3. Define `SpeakerClusterRecord`, `SpeakerClusterLabelRevision`, `SpeakerClusterAliasRevision`, and compact `SpeakerCluster` projection. A record is session-scoped and owns an opaque `speaker-<uuid>` ID. Add `SpeakerClusterCreated`, `SpeakerClusterLabelRevisionRecorded`, `SpeakerClusterAliasRevisionRecorded`, and `TranscriptSpeakerReassigned` to `AuditKind`.
4. Re-run the focused test; it must pass.
5. Commit with `feat: define anonymous speaker catalog records`.

## Task 2: Persist and verify catalog revisions

**Files:**
- Modify: `src-tauri/src/audit/store.rs`
- Test: `src-tauri/src/audit/store.rs`

1. Write failing tests for atomic cluster creation with its initial generated label and audit event; reopen behavior; bad kind/run/causation/payload bindings; parent gaps; cross-session/self/cyclic aliases; duplicate bindings; and tampered rows or event hashes. Assert legacy transcript rows with `speaker-1` still verify without catalog data.
2. Run `cargo test --manifest-path src-tauri/Cargo.toml --offline audit::store::tests --lib`; it should fail until catalog storage exists.
3. Add `speaker_clusters`, `speaker_cluster_label_revisions`, and `speaker_cluster_alias_revisions`, immutable triggers, unique revision chains, indexes, transactional append/list methods, and `AuditStore::verify()` bindings. Do not add foreign keys or backfill legacy cluster strings.
4. Re-run the focused suite; it must pass.
5. Commit with `feat: persist audited speaker catalog revisions`.

## Task 3: Add conflict-safe single-span corrections

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/state.rs`
- Test: `src-tauri/src/commands.rs`

1. Write failing tests for create/list/rename cluster and reassignment of a current final span. Confirm it appends one `UserEdited` revision plus one `TranscriptSpeakerReassigned` event while preserving text, timing, and model. Verify stale, partial, unknown, cross-session, and aliased-target cases leave timeline, SQLite, and audit length unchanged.
2. Run `cargo test --manifest-path src-tauri/Cargo.toml --offline state::tests commands::tests --lib`; it should fail before typed authority exists.
3. Add typed Tauri inputs for session ID, cluster ID, expected revision, logical span ID, and nullable target. `AppState` loads the durable current head, validates an active canonical target, makes a dedicated edit-time audit event, commits event plus revision atomically, then updates the in-memory timeline.
4. Re-run the focused suite; it must pass.
5. Commit with `feat: audit manual speaker reassignment`.

## Task 4: Add reversible merge and explicit split bundles

**Files:**
- Modify: `src-tauri/src/audit/store.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/audit/store.rs`
- Test: `src-tauri/src/state.rs`

1. Write failing tests for aliasing active source to target without rewriting history, clearing an alias by revision, and split creating a new anonymous cluster before moving only explicitly selected final heads. Inject a failure and verify full rollback.
2. Run `cargo test --manifest-path src-tauri/Cargo.toml --offline audit::store::tests state::tests --lib`; it should fail before bundle support.
3. Add a deterministic bundle transaction with cluster/alias events and individual transcript reassignment events/revisions. Preserve the `timelines -> audit_trail -> audit_store` lock order. Merge is an alias; split is create plus explicit reassignments; neither deletes or overwrites history.
4. Re-run the focused suite; it must pass.
5. Commit with `feat: add reversible speaker merge and split`.

## Task 5: Add the flat manual-correction workspace

**Files:**
- Create: `src/components/SpeakerManager.vue`
- Create: `src/components/SpeakerManager.spec.ts`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/lib/wordCovenantApi.spec.ts`
- Modify: `src/stores/session.ts`
- Modify: `src/stores/session.spec.ts`
- Modify: `src/components/TimelinePanel.vue`
- Modify: `src/components/TimelinePanel.spec.ts`
- Modify: `src/App.vue`
- Modify: `src/assets/main.css`
- Test: `src/App.spec.ts`

1. Write failing tests for opaque-ID label resolution, missing catalog display as `未归类`, a user label as manual metadata, selection references containing current revision, camelCase command input, and no optimistic update on error.
2. Run `pnpm test --run src/components/TimelinePanel.spec.ts src/components/SpeakerManager.spec.ts src/stores/session.spec.ts src/lib/wordCovenantApi.spec.ts`; it should fail until the manager exists.
3. Retain the three-column workspace. Open a compact inspector beside the timeline on wide displays and above it on narrow displays. Use flat white/gray surfaces, no nested cards, no fabricated confidence/identity, and red only for errors. Use real buttons, labelled checkboxes, `role="alert"`, and durable command results only.
4. Re-run the focused frontend suite; it must pass.
5. Commit with `feat: add manual speaker correction workspace`.

## Task 6: Verify and document boundaries

**Files:**
- Modify: `docs/plans/2026-08-07-word-covenant-roadmap.md`
- Modify: `README.md`
- Test: existing Rust and frontend suites

1. Mark manual anonymous correction as implemented only after verification. State that automatic diarization, embeddings, voice profiles, confidence, and identity claims remain unavailable.
2. Run the offline matrix:

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

Expected: all checks pass without downloading a model, creating a network client, exposing PCM, or changing default egress denial.

3. Commit with `docs: define manual speaker correction acceptance`.
