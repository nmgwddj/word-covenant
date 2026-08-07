# WordCovenant Product Roadmap and Execution Plan

> **For Codex:** Execute this plan in small, tested changes. Do not add an outbound client, cloud model, auto-download, arbitrary hook, or arbitrary shell execution outside the explicit milestone that permits it.

**Goal:** Deliver a macOS-first, local-first conversation record application whose audio, transcript, speaker data, and Agent context remain on-device unless the user visibly enables a narrowly scoped outbound action.

**Architecture:** WordCovenant is a modular Tauri desktop application. The Rust process is the sole owner of microphone access, capture time, storage, policy decisions, credentials, model execution, and tool execution. The Vue WebView only renders projections and sends typed intents. Every Agent, skill, hook, model, transcript, and external response may create an action proposal but cannot cause a side effect without the Rust policy and approval path.

**Tech Stack:** Tauri 2, Rust, Vue 3, TypeScript, Pinia, SQLite/FTS5, macOS Keychain, CoreAudio/CPAL, Metal-accelerated local ASR, ONNX Runtime, JSON Schema, `tracing`, and native macOS signing/notarization tooling.

---

## Product Contract

### Privacy and authority invariants

1. Recording is visible while active. There is no hidden, automatic room recording mode.
2. Audio, transcript, derived speaker embeddings, and Agent context stay local by default.
3. Egress is denied at startup and after every application restart. A network request needs all three gates below:
   - the visible session-level **Allow outbound access** switch is on;
   - the user has approved the exact built-in tool/profile, HTTPS origin, data categories, and duration;
   - the Rust executor re-evaluates the policy immediately before opening a connection.
4. Turning off the visible switch instantly blocks all future egress. Revoking a profile does the same. Neither one deletes local records.
5. WebView CSP/capabilities do not grant network or shell access. A future Rust HTTP client is created only after a successful policy decision.
6. Agent output is typed `PlanV1` data, never executable text. Arbitrary shell commands, arbitrary URLs, native sidecars, and generic MCP tools are not MVP capabilities.
7. Diarization initially assigns anonymous clusters. A display name or voiceprint profile is explicit user data, not an identity claim inferred from ambient audio.
8. The audit chain makes local changes detectable; it does not prove legal enforceability or non-repudiation.

### MVP definition

The first usable release is a visible local recorder that lets a macOS user start/stop capture, see final Chinese transcript spans with capture-time timestamps and anonymous speaker clusters, search/correct those spans, and manually trigger a local Agent action. It works offline after the user imports local models. It shows all recording, model, egress, approval, gap, and action decisions in local history.

Not MVP: automatic human identification from a voiceprint, ambient/background recording, overlap separation guarantees, cloud transcription, automatic model downloads, unattended external actions, generic plugins, legal-contract claims, or a cross-device sync service.

## Current Baseline and Immediate Work

| Status | Work item | Evidence / next exit gate |
| --- | --- | --- |
| Done | M0 application shell | WordCovenant branding, no shell/network Tauri capability, CSP without wildcard connect origins |
| Done | M0 local policy and audit core | Default-deny exact-origin approval checks and SQLite SHA-256 audit chain have unit coverage |
| Done | M0 workspace | Visible local-only/recording state, synthetic browser preview, typed Tauri command path |
| Done | M0 audio boundary | Sample clock, bounded queue, gap events, test source, macOS callback boundary |
| Done | M0.1 explicit egress control | Session-only Rust master gate, visible confirmation/disable UI, and regression coverage; no HTTP client exists |
| Done | M1 lifecycle foundation | Typed permission/recording/interruption state machine behind the macOS callback boundary |
| Pending | M1 native input adapter | Add a real CoreAudio/CPAL stream, microphone permission integration, and manual device validation; no ASR/model loading yet |

## Target Architecture

```mermaid
flowchart LR
  MIC["macOS input device"] --> CAP["Rust capture adapter\nclock + bounded PCM queue"]
  CAP --> PIPE["Local pipeline\nresample, VAD, ASR, diarization"]
  PIPE --> DB["Encrypted local records\nSQLite / audio chunks"]
  DB --> UI["Vue timeline, search, corrections"]
  UI --> TRIGGER["Manual Agent trigger"]
  TRIGGER --> PLANNER["Local / selected Planner\nreturns PlanV1 only"]
  PLANNER --> POLICY["Rust Policy Engine\nthree egress gates"]
  POLICY --> TOOL["Tool broker\nTTS / notification / HTTP profile"]
  TOOL --> AUDIT["Append-only audit chain"]
  PIPE --> AUDIT
  POLICY --> AUDIT
  AUDIT --> UI
```

### Data and time model

Every capture-derived record carries `session_id`, a monotonic capture range, a wall-clock anchor, a sample rate, a revision number, and model versions. CoreAudio host time / `mach_continuous_time` becomes the source of truth once native capture lands; browser `Date.now()` is display metadata only. Device changes, sleep, source loss, and queue overflow become explicit gap events rather than fabricated continuous time.

The minimum final transcript projection is:

```text
TranscriptSpan {
  id, session_id,
  capture_start_ns, capture_end_ns,
  wall_clock_start,
  speaker_cluster_id: Option<String>,
  text, is_final, revision,
  asr_model_version, diarization_model_version,
  overlap, confidence
}
```

## Delivery Milestones

### M0.1: Visible Egress Gate (completed)

**Outcome:** A local approval record alone is insufficient. A clearly visible, session-only user switch is required before policy can allow outbound behavior.

**Implementation tasks:**

1. Backend gate
   - Modify `src-tauri/src/policy/egress.rs`, `src-tauri/src/state.rs`, `src-tauri/src/commands.rs`, and `src-tauri/src/lib.rs`.
   - Add session-level egress state with default `Disabled`; do not persist an enabled state across app launch.
   - Require that state to be `Enabled` before matching approvals are considered.
   - Add typed `set_egress_enabled` and extend `PrivacyStatus` with `egress_enabled`.
   - Audit enabled/disabled transitions without storing sensitive content.
   - Test: approval plus disabled switch denies; enable plus matching approval allows; disabling again denies; restart defaults to deny.

2. Visible consent surface
   - Modify `src/types.ts`, `src/lib/wordCovenantApi.ts`, `src/stores/privacy.ts`, `src/App.vue`, and add a focused component for the switch/confirmation.
   - The enable confirmation names the fact that data may leave the device and specifies that the switch only permits separately approved profiles.
   - Display active profile count and a direct Disable action at all times. Browser preview remains local fake data only.
   - Test: no one-click implicit enable, cancel leaves disabled, enable refreshes status, disable is immediate.

**Exit gate:** `cargo test`, frontend tests/type-check, and a browser/Tauri UI inspection demonstrate that an approved profile is still denied until a user visibly enables the session switch. No outbound HTTP request is introduced in this milestone.

### M1.0: Capture Lifecycle Foundation (completed)

**Outcome:** The application has a deterministic, UI-safe state model for `Idle`, `AwaitingPermission`, `Recording`, `Interrupted`, and `Failed` before it touches real microphone hardware.

**Delivered:** `CaptureLifecycle` accepts only valid transitions from the existing macOS callback boundary, tracks the selected device and transition time, represents unavailable-device/closed-queue failures as interruptions, and rejects stale-device/invalid transitions without mutation. It intentionally does not request macOS permission or claim that a CoreAudio stream is running.

**Verification:** Deterministic Rust tests cover permission-resolution state, start, device change, interruption, normal stop, device mismatch, and external failure.

### M1: Local Capture Vertical Slice

**Outcome:** A real selected input device produces a local PCM stream, clear recording status, level meter, and durable gap-aware session metadata.

**Tasks:**

1. Add a `CaptureService` lifecycle (`Idle`, `AwaitingPermission`, `Recording`, `Interrupted`, `Failed`) and bind start/stop commands to it.
2. Add `src-tauri/src/audio/cpal_input.rs` behind the current callback boundary. The real-time callback may only enqueue bounded PCM; it cannot run inference, write storage, update UI, or await work.
3. Add `NSMicrophoneUsageDescription`, explicit permission resolution, input-device selection, and source-loss recovery. A restarted source starts a new clock segment, never silently reuses sample offsets.
4. Throttle compact meter and lifecycle events to the WebView; PCM never crosses IPC.
5. Persist session anchors, selected device identifier, sample format, and capture gaps in SQLite.
6. Add microphone permission failure UI and a non-recording recovery state.
7. Add a manual macOS test script covering deny/allow permission, device removal, sleep/wake, and queue overload.

**Acceptance targets:** start/stop affects a real microphone; visible indicator appears before capture; source loss is recorded as a gap; one hour of local capture has no unbounded queue/memory growth; no network dependency is introduced.

### M2: Offline Speech Understanding

**Outcome:** Explicitly installed local models produce revisioned Chinese transcript spans without automatic downloads.

**Tasks:**

1. Create a local model registry containing file path, SHA-256, model card/license acknowledgement, input format, size, and model version. Import copies or registers a local user-selected file only.
2. Add `ModelProvider` traits for VAD, ASR, and embedding inference. Test each with deterministic fixture adapters before native runtimes.
3. Add 48 kHz-to-16 kHz resampling, VAD pre-roll/hangover, rolling windows, and finalization rules.
4. Integrate `whisper.cpp`/Metal as the first ASR adapter. Emit partial spans for display and immutable final revisions for Agent input.
5. Store transcript spans and FTS5 search projections locally. Corrections create a revision, never overwrite original model output.
6. Build a consented fixture benchmark: Chinese CER/WER, p95 partial/final latency, real-time factor, RAM, thermal/energy, and model import time.

**Acceptance targets:** offline after model import; Agent only receives final spans; model/file/license provenance is visible; quality claims are benchmarked rather than assumed.

### M3: Speaker Workflow

**Outcome:** The timeline groups sufficiently clear non-overlapping speech into anonymous, correctable clusters.

**Tasks:**

1. Add local speaker embedding adapter and online cosine-similarity clustering with a calibrated ambiguity threshold.
2. Persist an embedding reference/hash separately from transcript projection; encrypt or delete it with the cluster.
3. Show `Speaker 1/2/...`, confidence, overlap/uncertain labels, and manual rename/merge/split controls.
4. Require an explicit consent flow for mapping an anonymous cluster to a profile. Support deletion, re-enrolment, and all related audit events.
5. Benchmark diarization error rate and wrong-assignment rate on consented, licensed fixtures; test ambiguity paths.

**Acceptance targets:** no automatic personal-name assertion; overlapping/uncertain speech remains visibly uncertain; edits are revisioned and reversible.

### M4: Agent Actions and Controlled Tools

**Outcome:** A manually invoked Agent can select approved local context, propose typed actions, and execute only after the policy/approval path.

**Tasks:**

1. Add `ContextSelector` with explicit session/span selection and redaction preview. Default context is final local spans only.
2. Add a `Planner` trait and strict JSON `PlanV1` validator. Local planner adapters come first; a cloud planner itself is an outbound profile and cannot bypass egress controls.
3. Add `ToolBroker` with local TTS and notification executors. TTS uses fixed native APIs/arguments, not a shell string; playback pauses transcript-triggered actions to prevent feedback loops.
4. Implement named HTTP profiles with fixed HTTPS origin, method, request/response schemas, byte limits, timeout, retry rules, idempotency key, and Keychain credential references.
5. Add per-action confirmation UI showing tool/version, destination, data categories, redacted payload summary, scope/duration, and audit run ID.
6. Wire an actual HTTP client only here, after policy revalidation immediately before sending. Treat remote responses as untrusted data.

**Acceptance targets:** every action can be traced from selected spans through plan, approval, policy, tool result, and audit record; switch-off/revocation wins over a queued action; no model can mint permissions.

### M5: Declarative Skills and Constrained Hooks

**Outcome:** Users can add capabilities without giving arbitrary code unreviewed authority.

**Tasks:**

1. Define declarative skills as `manifest.json`, `SKILL.md`, JSON Schema, declared data categories, and named built-in tools. Validate content hash and schema before loading.
2. Provide only proposal-producing hook events: `on_transcript_finalized`, `on_manual_trigger`, `before_tool_call`, `after_tool_call`, `on_run_finished`.
3. Add causation IDs, recursion limits, idempotency keys, output-size limits, and audit records for every hook proposal.
4. Package signed skills (`.wcpkg`, Ed25519) and show signer/version/hash. Signature is provenance, not permission.
5. Only after the declarative model is reliable, evaluate an opt-in Wasmtime hook sandbox with no filesystem, environment, network, or preopened directories and strict fuel/memory/time limits.

**Acceptance targets:** a skill can propose but cannot execute; unsigned/untrusted input cannot escape declared schemas; hooks cannot loop indefinitely or receive unfiltered raw context.

### M6: Data Protection, Evidence, and Release Operations

**Outcome:** The application supports durable local ownership, recovery, carefully scoped export, and macOS release quality.

**Tasks:**

1. Put database/audio encryption keys in Keychain; add SQLCipher or field-level AEAD after an explicit threat-model review.
2. Add raw-audio opt-in, encrypted chunk storage, retention schedules, deletion proof/audit entries, and storage quota UX.
3. Create an export manifest with record hashes, audit-chain verification result, model/tool versions, redaction options, and detached signatures where appropriate. Do not call it legal proof.
4. Add backup/restore, integrity verification, corruption handling, and migrations.
5. Add structured local `tracing` with secret/transcript redaction; opt-in diagnostics are a separate egress profile.
6. Sign, notarize, sandbox, and test the macOS bundle; publish a clear microphone/biometric/privacy notice and release checklist.

**Acceptance targets:** a user can inspect, export, retain, and delete their local data; restore detects tampering/corruption; release builds pass signed/notarized macOS smoke tests.

## Dependencies and Parallelization

```text
M0.1 egress gate -----------+------------------> M4 HTTP profiles
M1 capture lifecycle -------+--> M2 ASR --------+--> M3 clustering
M2 model registry ----------+
M2 final transcript events -+--> M4 Agent context --> M5 declarative skills
M0 audit core ------------------------------------> M4 / M5 / M6
M1 permission/release work ----------------------> M6 notarization
```

The safe parallel units are UI-only policy projections, pure Rust policy tests, capture adapter code, fixture/benchmark tooling, and documentation. The following must stay sequenced: actual HTTP client after M0.1+M4 policy/approval paths; voiceprint naming after anonymous clustering; executable hooks after declarative skills; encryption migration after a threat-model decision.

## Non-Functional Gates

| Area | Initial target | How it is checked |
| --- | --- | --- |
| Privacy | Zero egress with switch off | Unit/integration test plus local network monitor during manual test |
| Capture time | No fabricated continuity across input gaps | Clock/gap unit tests and sleep/device manual scenarios |
| UI | Recording and egress state always visible | Component tests, desktop screenshot/manual QA |
| Audio latency | Benchmark before promising a target; record p95 and RTF per model/device | Versioned local benchmark report |
| Reliability | No unbounded audio queue; recoverable device loss | Stress fixture, queue-overrun and unplug tests |
| Data integrity | Audit chain verifies on open | Tamper/reopen tests |
| Accessibility | Keyboard usable controls and programmatic labels | Component tests plus manual VoiceOver pass |
| Release | Signed/notarized macOS bundle | CI/release checklist and clean-machine smoke test |

## Risk Register

| Risk | Mitigation | Release decision |
| --- | --- | --- |
| Recording/biometric law differs by place | Visible recording, anonymous clusters by default, consent/revocation/deletion, jurisdiction-specific legal review | Block broad release until legal review |
| Single mic cannot separate overlap reliably | Label uncertainty/overlap; do not infer a speaker | Never advertise guaranteed attribution |
| Model license or weights restrict use | Record model provenance/license at import | Block model from production registry |
| UI switch is bypassed by backend/plugin | Central Rust gate, static deny-by-default capabilities, regression tests | Block release on any bypass |
| TTS feeds back into transcription/Agent | Pause triggering during local playback; later add AEC/source tagging | Limit MVP behavior until tested |
| Native sidecars inherit user rights | Exclude from normal product; developer mode only after design review | Do not ship in MVP |

## Verification Matrix

Run after each change set:

```sh
pnpm test --run
pnpm type-check
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

For M1 onward, add a manual macOS run before marking a milestone complete. Record machine model, macOS version, input device, permission result, expected/observed event sequence, and whether any local network monitor observed egress. Do not mark an untested hardware/model path complete based solely on compilation.
