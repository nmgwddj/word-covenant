# WordCovenant Local-First MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a macOS-first WordCovenant desktop MVP that records an explicitly started local session, presents timestamped transcript events and speaker clusters, and can run user-triggered, policy-governed actions without any network egress unless the user explicitly enables it.

**Architecture:** Keep the product a local-first modular monolith. The Tauri Rust process owns audio, clocks, persistence, policy, audit events, and tool execution; the Vue WebView renders projections and requests typed commands. Every model, planner, skill, hook, and tool can only propose an action. A Rust policy gate validates capability, consent, and data scope before a side effect occurs.

**Tech Stack:** Tauri 2, Vue 3, TypeScript, Vite, Tailwind 4, Pinia, Rust, Tokio, Serde, SQLite/SQLCipher, macOS Keychain, CoreAudio/CPAL, ONNX Runtime, whisper.cpp/Metal, JSON Schema, tracing.

---

## Product Contract

### Non-negotiable privacy rules

1. Audio, transcript, speaker embedding, and all derived context remain local by default.
2. Network egress is denied by default in both application policy and Tauri capability/CSP configuration.
3. A user must explicitly enable a named tool/version and its exact destination before any outbound request. The approval records data categories, scope, duration, and revocation state.
4. A model, skill, hook, transcript, or external response cannot grant itself network or system authority.
5. The UI always shows recording state, local-only state, and a visible active-egress indicator.
6. Raw audio retention is opt-in, encrypted, time-bounded, and deletable. The initial default is transcript-only storage.
7. Speaker diarization starts as anonymous clusters (`Speaker 1`, `Speaker 2`). Mapping a cluster to a person is explicit user data, not an automatic identity assertion.

### MVP success criteria

- A user can start and stop a visibly indicated capture session on macOS.
- The Rust layer persists immutable, monotonic-time `TranscriptSpan` events and renders them in a chronological Vue timeline.
- A session and its events remain usable without an internet connection.
- User-triggered agent planning produces a typed `PlanV1`, never an executable shell string.
- A built-in local action can be executed and audited. A network action is rejected until a persisted explicit approval exists.
- Tests prove the default-deny egress policy, ordered audit chain, timeline projection, and UI status states.

### MVP exclusions

- Automatic naming of people from voiceprints.
- Claiming that recordings are legally binding contracts or non-repudiation evidence.
- Generic native hooks, arbitrary shell commands, unrestricted MCP servers, or unsigned plugin code.
- Unattended network calls, cloud transcription, background data sync, and model auto-downloads.
- Reliable overlapping-speech separation from a single Mac microphone.

## Data Model and Event Flow

```text
CoreAudio input -> sample clock -> bounded PCM ring buffer -> VAD / ASR / speaker adapter
  -> TranscriptSpan (revisioned, stable/final state) -> append-only audit event
  -> SQLite projection -> Tauri query command -> Vue timeline

Manual trigger -> context selector -> planner -> PlanV1 JSON validation
  -> policy and consent check -> built-in tool broker -> audit outcome -> UI action history
```

### Time model

Store `session_wall_clock_anchor`, `monotonic_capture_ns`, `sample_offset`, and `sample_rate` for every event. Use CoreAudio host time or `mach_continuous_time` as the capture source when the native implementation arrives. Browser timestamps are presentation metadata only. Persist device-change, sleep, and dropped-audio `gap` events instead of pretending that time was continuous.

### Minimum records

```rust
TranscriptSpan {
    id: Uuid,
    session_id: Uuid,
    capture_start_ns: u64,
    capture_end_ns: u64,
    speaker_cluster_id: Option<String>,
    text: String,
    is_final: bool,
    revision: u32,
    source: TranscriptSource,
}

AuditEvent {
    id: Uuid,
    run_id: Option<Uuid>,
    causation_id: Option<Uuid>,
    kind: AuditKind,
    monotonic_ns: u64,
    wall_clock: DateTime<Utc>,
    payload_hash: String,
    previous_hash: Option<String>,
    hash: String,
}
```

## Delivery Sequence

| Milestone | Scope | Exit gate |
| --- | --- | --- |
| M0: Trustworthy shell | Brand, local-only policy, typed commands, audit data, workspace UI | No network path works without explicit approval; tests pass |
| M1: Local capture vertical slice | CoreAudio capture adapter, session clock, waveform levels, local storage | Start/stop session works on a real Mac and produces gap-aware events |
| M2: Speech understanding | User-installed local VAD/ASR model adapters and revisioned transcript spans | Offline Chinese speech reaches accepted latency/accuracy targets on consented fixtures |
| M3: Speaker workflow | Anonymous clustering, manual merge/rename, confidence and corrections | Incorrect identity is never asserted automatically |
| M4: Agent actions | Context selection, PlanV1, approval UX, local TTS and opted-in HTTP profiles | Every action is explainable and replayable from audit data |
| M5: Extensibility | Signed declarative skills, then constrained WASM hooks | Plugins cannot bypass policy or read undeclared data |
| M6: Evidence and operations | Encrypted audio option, export hash manifests, backup/restore, notarized release | Product makes auditable claims without overclaiming legal effect |

## Task 1: Rebrand and Lock Down the Application Shell

**Files:**
- Modify: `package.json`
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/tauri.conf.json`
- Modify: `index.html`
- Create: `src-tauri/capabilities/default.json`
- Test: `tests/unit/product-metadata.test.ts`

**Step 1: Write the failing metadata test.**

Assert that the visible title is `WordCovenant`, the package names are `word-covenant`, and the default outbound policy is disabled.

**Step 2: Run the test to verify it fails.**

Run: `pnpm test --run tests/unit/product-metadata.test.ts`

Expected: FAIL because the generated template exposes `tauri-app`.

**Step 3: Replace template metadata and default permissions.**

Set product name, macOS bundle identifier, accessible window title, and slogan metadata. Remove the shell plugin from the default capability surface. Configure CSP to permit only Tauri IPC and local asset protocols; do not include wildcard `connect-src` entries.

**Step 4: Re-run focused tests and build checks.**

Run: `pnpm test --run tests/unit/product-metadata.test.ts && pnpm type-check && pnpm check`

Expected: PASS.

**Step 5: Commit.**

```bash
git add package.json index.html src-tauri/Cargo.toml src-tauri/tauri.conf.json src-tauri/capabilities/default.json tests/unit/product-metadata.test.ts
git commit -m "feat: brand and lock down WordCovenant shell"
```

## Task 2: Establish Rust Domain Contracts and Default-Deny Egress Policy

**Files:**
- Create: `src-tauri/src/domain/mod.rs`
- Create: `src-tauri/src/domain/session.rs`
- Create: `src-tauri/src/domain/transcript.rs`
- Create: `src-tauri/src/domain/agent.rs`
- Create: `src-tauri/src/policy/mod.rs`
- Create: `src-tauri/src/policy/egress.rs`
- Modify: `src-tauri/src/lib.rs`

**Step 1: Write failing unit tests.**

Test that a new `EgressPolicy` rejects all network tools, a named approved tool and exact origin can execute, expired approval is rejected, and an action plan cannot contain arbitrary command text.

**Step 2: Run the Rust tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml policy:: -- --nocapture`

Expected: FAIL because the modules do not exist.

**Step 3: Add serializable, UI-safe contracts.**

Define `CaptureSession`, `TranscriptSpan`, `SpeakerCluster`, `PlanV1`, `ActionProposal`, `ToolPermission`, `EgressApproval`, `PolicyDecision`, and `PolicyReason`. Use a closed `ToolKind` enum, schema-backed input payloads, origin normalization, expiry, scopes, and no shell-command variant.

**Step 4: Implement minimal policy evaluation.**

Return `DeniedByDefault`, `MissingExplicitApproval`, `OriginMismatch`, `ApprovalExpired`, or `Allowed`. Keep all policy evaluation deterministic and free of I/O.

**Step 5: Re-run tests and format.**

Run: `cargo fmt --manifest-path src-tauri/Cargo.toml --check && cargo test --manifest-path src-tauri/Cargo.toml policy::`

Expected: PASS.

**Step 6: Commit.**

```bash
git add src-tauri/src/domain src-tauri/src/policy src-tauri/src/lib.rs
git commit -m "feat: add local-first policy contracts"
```

## Task 3: Add Append-Only Local Audit Store

**Files:**
- Create: `src-tauri/src/audit/mod.rs`
- Create: `src-tauri/src/audit/hash_chain.rs`
- Create: `src-tauri/src/audit/store.rs`
- Modify: `src-tauri/Cargo.toml`
- Test: Rust module tests in `src-tauri/src/audit/*.rs`

**Step 1: Write failing tests for chain ordering and tamper detection.**

Create three events; assert each hash includes the previous hash. Modify serialized payload data and assert verification fails. Assert that sensitive body text is represented by a hash/reference rather than copied into audit metadata.

**Step 2: Add minimal dependencies.**

Add `chrono`, `sha2`, `uuid`, and `rusqlite` with a local bundled SQLite configuration. Do not add SQLCipher until the base store is tested; introduce encryption as a dedicated migration task.

**Step 3: Implement an in-memory store first, then SQLite repository.**

Expose `append`, `verify`, `list_for_session`, and `list_for_run`. Write migration SQL inside the Rust module so every application database has the same schema.

**Step 4: Run Rust tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audit::`

Expected: PASS.

**Step 5: Commit.**

```bash
git add src-tauri/Cargo.toml src-tauri/src/audit
git commit -m "feat: add append-only local audit store"
```

## Task 4: Build the Audio Capture Boundary Without Loading Models

**Files:**
- Create: `src-tauri/src/audio/mod.rs`
- Create: `src-tauri/src/audio/clock.rs`
- Create: `src-tauri/src/audio/capture.rs`
- Create: `src-tauri/src/audio/test_source.rs`
- Modify: `src-tauri/Cargo.toml`
- Test: Rust module tests in `src-tauri/src/audio/*.rs`

**Step 1: Write failing clock tests.**

Verify conversion from `(sample_offset, sample_rate)` to nanoseconds, monotonic span ordering, and explicit gap generation after a device restart.

**Step 2: Implement a test source and bounded channel.**

The capture callback contract only writes PCM metadata into a bounded queue. It must not call the UI, allocate arbitrary buffers, run model inference, or await I/O.

**Step 3: Implement the macOS capture adapter behind a feature boundary.**

Use `cpal` for the initial CoreAudio input adapter. Capture selection, raw PCM, and host-clock precision remain private Rust implementation details. The Vue layer only receives throttled meter/session events.

**Step 4: Run tests and compile check.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audio:: && pnpm check`

Expected: PASS.

**Step 5: Commit.**

```bash
git add src-tauri/Cargo.toml src-tauri/src/audio
git commit -m "feat: add local capture boundary and session clock"
```

## Task 5: Expose Typed Tauri Commands and Application State

**Files:**
- Create: `src-tauri/src/state.rs`
- Create: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: Rust module tests in `src-tauri/src/commands.rs`

**Step 1: Define commands.**

Implement `get_privacy_status`, `start_session`, `stop_session`, `list_timeline`, `create_egress_approval`, `revoke_egress_approval`, `propose_action`, and `execute_action`.

**Step 2: Write failing command tests.**

Assert an unapproved network `ActionProposal` is denied and audited; assert a local notification/TTS proposal follows policy; assert timeline queries only return their session records.

**Step 3: Register managed state and commands.**

Use `tauri::State` with synchronization scoped to the application lifecycle. Emit only compact UI projection events, never PCM, raw audio, secret values, or full network response bodies.

**Step 4: Run tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml commands::`

Expected: PASS.

**Step 5: Commit.**

```bash
git add src-tauri/src/state.rs src-tauri/src/commands.rs src-tauri/src/lib.rs
git commit -m "feat: expose policy-governed Tauri commands"
```

## Task 6: Replace the Template UI With the Recording Workspace

**Files:**
- Create: `src/types.ts`
- Create: `src/lib/wordCovenantApi.ts`
- Create: `src/stores/session.ts`
- Create: `src/stores/privacy.ts`
- Create: `src/components/RecordingControl.vue`
- Create: `src/components/TimelinePanel.vue`
- Create: `src/components/PrivacyStatus.vue`
- Create: `src/components/AgentActionPanel.vue`
- Modify: `src/App.vue`
- Modify: `src/assets/main.css`
- Test: `src/components/*.spec.ts`

**Step 1: Write component tests.**

Test visible local-only status, a persistent recording indicator, disabled external action when egress is off, timeline chronological ordering, and the confirmation state for a network permission.

**Step 2: Create a typed IPC client.**

All `invoke` names and payloads live in one module. Browser development mode uses a deterministic local fake; it must never fall back to a real HTTP API.

**Step 3: Implement Pinia stores and the workspace.**

Use a practical desktop layout: session controls and level meter at the top, transcript timeline as the dominant work area, and action/audit history at the side. Include speaker chips, timestamp navigation, recording state, and an always-visible `Local only` status.

**Step 4: Run focused frontend tests and type checking.**

Run: `pnpm test --run src/components src/stores && pnpm type-check`

Expected: PASS.

**Step 5: Commit.**

```bash
git add src src/assets/main.css
git commit -m "feat: add local-first recording workspace"
```

## Task 7: Connect the Vertical Slice and Test It End to End

**Files:**
- Modify: `src-tauri/src/lib.rs`
- Modify: `src/App.vue`
- Create: `tests/unit/word-covenant-api.test.ts`
- Modify: `README.md`

**Step 1: Wire start/stop, demo transcript events, and action proposal flow.**

Use a local test capture source in development until the microphone adapter is explicitly selected. Label synthetic values in the UI so they can never be confused with a real recording.

**Step 2: Add integration checks.**

Verify: start session -> receive ordered local timeline entries -> manual trigger creates a plan -> external HTTP action is denied -> approval enables only the exact origin -> revoke denies it again.

**Step 3: Run the complete MVP verification set.**

Run:

```bash
pnpm install --frozen-lockfile
pnpm test --run
pnpm type-check
pnpm check
cargo test --manifest-path src-tauri/Cargo.toml
pnpm build
```

Expected: all commands pass. Perform a macOS manual run to verify microphone permission text, the visible recording state, and no network request before an explicit approval.

**Step 4: Commit.**

```bash
git add README.md src src-tauri tests
git commit -m "feat: deliver local-first WordCovenant vertical slice"
```

## Task 8: M2 Local Speech Model Adapters

**Files:**
- Create: `src-tauri/src/inference/mod.rs`
- Create: `src-tauri/src/inference/vad.rs`
- Create: `src-tauri/src/inference/asr.rs`
- Create: `src-tauri/src/inference/model_registry.rs`
- Create: `src-tauri/src/inference/mock.rs`
- Modify: `src-tauri/src/audio/mod.rs`
- Modify: `src-tauri/src/domain/transcript.rs`

**Steps:**

1. Create an `InferenceEngine` trait and prove mock output becomes revisioned transcript events.
2. Add an explicit local model install/import flow with file hashes, license acknowledgement, disk estimate, and no automatic download.
3. Add Silero VAD through ONNX Runtime and validate 16 kHz segmentation with licensed fixtures.
4. Add a `whisper.cpp` Metal adapter behind the same trait; emit temporary and final spans separately.
5. Benchmark latency, memory, CPU, battery impact, Chinese CER/WER, and dropped-audio rate on consented recordings.

**Exit gate:** No model byte leaves the device; model selection, model version, and finalization revision appear in the event/audit record.

## Task 9: M3 Speaker Cluster Workflow

**Files:**
- Create: `src-tauri/src/diarization/mod.rs`
- Create: `src-tauri/src/diarization/embedding.rs`
- Create: `src-tauri/src/diarization/clustering.rs`
- Create: `src/components/SpeakerManager.vue`
- Modify: `src/components/TimelinePanel.vue`

**Steps:**

1. Build test fixtures for anonymous embeddings; prove thresholding and manual merges are deterministic.
2. Add an ONNX ECAPA-TDNN/WeSpeaker embedding adapter only after model-license review.
3. Display anonymous cluster labels and confidence; mark short, overlapping, or noisy spans `uncertain`.
4. Implement rename/merge/split corrections as auditable events. Do not create automatic identity bindings.
5. Make voice profile enrollment a separately consented future capability, not an MVP toggle.

**Exit gate:** The product never represents an inferred speaker identity as fact without user confirmation.

## Task 10: M4 Agent, Tool, and Consent System

**Files:**
- Create: `src-tauri/src/agent/context.rs`
- Create: `src-tauri/src/agent/planner.rs`
- Create: `src-tauri/src/tools/mod.rs`
- Create: `src-tauri/src/tools/local_tts.rs`
- Create: `src-tauri/src/tools/http_profile.rs`
- Create: `src/components/ActionApproval.vue`
- Modify: `src-tauri/src/policy/egress.rs`

**Steps:**

1. Select only final transcript spans and user-selected session context for planner input.
2. Require `PlanV1` JSON validation before policy evaluation. Planner output is an untrusted proposal.
3. Implement local TTS first with `AVSpeechSynthesizer`/a bounded native bridge. Pause trigger processing while TTS plays to prevent feedback loops.
4. Define named HTTP profiles with fixed origin, HTTP method, JSON Schema, byte limit, timeout, retries, and idempotency key.
5. Show a confirmation screen that names the tool/version, destination, data categories, permission duration, and revocation action.
6. Persist all plan, decision, approval, execution, response-hash, and failure events.

**Exit gate:** Network behavior is impossible unless an explicit, active, matching user approval is stored locally.

## Task 11: M5 Extension System

**Files:**
- Create: `docs/extensions/skill-manifest.schema.json`
- Create: `src-tauri/src/extensions/mod.rs`
- Create: `src-tauri/src/extensions/manifest.rs`
- Create: `src-tauri/src/extensions/validator.rs`
- Test: `src-tauri/src/extensions/*.rs`

**Steps:**

1. Support declarative `SKILL.md` plus `manifest.json` only.
2. Validate content hashes, declared tools, data needs, and JSON Schemas before registration.
3. Allow hooks only at `on_transcript_finalized`, `on_manual_trigger`, `before_tool_call`, `after_tool_call`, and `on_run_finished`.
4. Make hooks return annotations or action proposals; pass every proposal through the same policy engine.
5. Later, add signed packages and constrained WASM with no ambient filesystem, environment, or network access.

## Task 12: M6 Encryption, Release, and Operational Readiness

**Files:**
- Create: `src-tauri/src/security/key_store.rs`
- Create: `src-tauri/src/security/encryption.rs`
- Create: `docs/privacy.md`
- Create: `docs/threat-model.md`
- Modify: `README.md`
- Modify: GitHub release workflow files

**Steps:**

1. Store data keys in macOS Keychain and add migration-safe at-rest encryption.
2. Add retention policies, secure delete workflow, encrypted export, import, and backup recovery tests.
3. Generate a manifest of hashes and audit references for an export; document that it does not itself establish legal non-repudiation.
4. Add microphone permission strings, visible recording UX, Developer ID signing, notarization, and update signature verification.
5. Complete an external privacy, data protection, and model-license review before release.

## Parallel Execution Boundaries for the First Slice

| Workstream | Owner boundary | Files owned initially |
| --- | --- | --- |
| Rust policy and audit | Pure domain and audit modules | `src-tauri/src/domain/**`, `src-tauri/src/policy/**`, `src-tauri/src/audit/**` |
| Audio foundation | Capture contracts and deterministic sources | `src-tauri/src/audio/**` |
| Vue workspace | UI types, stores, components, styles | `src/types.ts`, `src/lib/**`, `src/stores/**`, `src/components/**`, `src/App.vue`, `src/assets/main.css` |
| Integration | Manifests, Tauri state/commands, test commands, build validation | `package.json`, `src-tauri/Cargo.toml`, `src-tauri/src/lib.rs`, `src-tauri/src/state.rs`, `src-tauri/src/commands.rs`, config files |

No workstream edits another workstream's owned files without an explicit handoff. The integration owner resolves dependency additions and command registration after reviewing each branch of work.

## Verification Matrix

| Risk | Proof |
| --- | --- |
| Unapproved egress | Unit test default denial, origin mismatch, expiry, revocation; manual network inspector check |
| Leaking raw audio | Architecture test that WebView command payloads contain no PCM; audit redaction test |
| Timestamp drift | Clock conversion, gap, and ordering unit tests; real capture comparison with CoreAudio host time |
| Unsafe agent execution | Plan schema and tool-policy tests; no shell tool in `ToolKind` |
| UI hides privacy status | Vue rendering test for recording and local-only states |
| Data tampering | Audit chain verification test and export manifest test |
| Model regressions | Fixed consented audio corpus with CER/WER, DER, latency, memory, and energy thresholds |

## Decision Log

- **ADR-001: Local-first default.** Approved by product direction. All egress requires user-visible, persisted explicit enablement.
- **ADR-002: Modular monolith.** Approved for MVP. There is no server dependency until explicitly requested for opt-in sync.
- **ADR-003: Anonymous speaker clusters first.** Avoids treating biometric inference as identity.
- **ADR-004: Typed plan and policy broker.** LLMs, skills, and hooks do not directly execute tools.
- **ADR-005: Audio and models stay out of the WebView.** Rust owns real-time resources and the UI receives small projections.
