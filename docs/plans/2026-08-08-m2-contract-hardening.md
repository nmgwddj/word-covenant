# M2 Contract Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every persisted M2 transcript and local-model record cryptographically bound to its audit event, private to the native process, and correctly recoverable after restart.

**Architecture:** Keep the existing local SQLite store and hash chain, but make typed record-to-event binding a store invariant. Persist only application-managed relative model paths, validate the resolved artifact at restore time, and expose a path-free model summary through Tauri IPC. Timeline hydration selects one recovered session and uses stored wall-clock values; ASR emission mapping keeps partial state transient while creating an initial durable revision for the first final emission.

**Tech Stack:** Rust, Tauri 2 commands, rusqlite/FTS5, SHA-256, Vue 3, Pinia, Vitest.

---

### Task 1: Bind audit events to durable records

**Files:**
- Modify: `src-tauri/src/audit/hash_chain.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Test: `src-tauri/src/audit/store.rs`

1. Add a typed helper on `AuditEvent` that recomputes the payload digest and proves it matches a supplied serializable payload.
2. Make transcript and local-model write methods reject an event with the wrong kind, record payload, session linkage, or capture endpoint.
3. Remove the public unaudited transcript-write path, then update immutability and lineage tests to write their matching events.
4. Add tests proving mismatched payloads and unaudited durable records cannot be committed, while a valid chain still verifies.
5. Run `cargo test --manifest-path src-tauri/Cargo.toml audit::store --lib`.

### Task 2: Keep model paths native-only and verify restored artifacts

**Files:**
- Modify: `src-tauri/src/inference/model_registry.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.spec.ts`
- Test: `src-tauri/src/inference/model_registry.rs`

1. Write tests that a persisted model path must be relative, stays under the model root, and resolves to a regular file with its recorded size and SHA-256.
2. Change imported/persisted model metadata to retain only the managed relative path; resolve its absolute path only inside the native registry.
3. Validate restored artifacts against the supplied managed root before indexing them, rejecting missing, replaced, symlinked, or hash-mismatched files.
4. Mark the native path as non-serializable and remove `filePath` from the WebView type contract and tests.
5. Ensure audit payloads contain provenance only, not local filesystem paths; run focused Rust and frontend tests.

### Task 3: Recover one timeline with persisted wall-clock timestamps

**Files:**
- Modify: `src-tauri/src/domain/transcript.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/stores/session.ts`
- Modify: `src/components/TimelinePanel.vue`
- Test: `src-tauri/src/state.rs`

1. Add a path-free timeline projection that includes immutable revision provenance and persisted wall-clock timing.
2. On startup, select the most recently persisted session rather than merging every historical session into the current view.
3. Render recovered entries from their wall-clock values whenever no live session clock is available.
4. Add a reopening test with two sessions that proves only the latest session is projected and its visible time is not calculated from zero.
5. Run `cargo test --manifest-path src-tauri/Cargo.toml state::tests --lib` and `pnpm test --run`.

### Task 4: Map ASR partial/final emissions without persisting partials

**Files:**
- Modify: `src-tauri/src/inference/asr.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Modify: `src-tauri/src/state.rs`
- Test: `src-tauri/src/inference/asr.rs`
- Test: `src-tauri/src/state.rs`

1. Add a Rust-only emission mapper keyed by session ID and ASR utterance key.
2. Keep partial emissions in the mapper's transient state and never pass them to SQLite or Agent context.
3. Turn the first final emission into durable revision 1 with the stable logical span ID allocated for its partial/final utterance, even when partials were disabled.
4. Use the persisted capture wall-clock anchor to build the final `TranscriptRevision` model provenance.
5. Add fixture coverage showing a partial plus revision-2 final produces one auditable durable final record with no partial FTS row.

### Task 5: Final verification and review

**Files:**
- Modify: `README.md` only if behavior wording changes

1. Run `cargo fmt --manifest-path src-tauri/Cargo.toml --check`.
2. Run `cargo test --manifest-path src-tauri/Cargo.toml` and `cargo check --manifest-path src-tauri/Cargo.toml --release`.
3. Run `pnpm test --run`, `pnpm type-check`, `pnpm build`, and `git diff --check`.
4. Reinspect the desktop model-import UI and its compact breakpoint.
5. Review the staged diff, commit the M2 batch, and push `main`.
