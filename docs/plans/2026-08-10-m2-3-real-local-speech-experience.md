# M2.3 Real Local Speech Experience Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let a macOS user explicitly import and select a local multilingual Whisper model, then record microphone speech, detect voice activity locally, persist final timestamped transcripts, and see them in the active timeline without any application egress.

**Architecture:** The existing CPAL ingress and bounded `CaptureDispatcher` remain the only native PCM consumer. A WebRTC VAD detector seals bounded speech windows, and a separately owned `whisper-rs`/whisper.cpp worker transcribes those windows using a verified, user-imported artifact. The state layer continues to atomically bind final output to SQLite and the audit chain; the WebView receives only projections and refreshes the durable timeline when final-result counters advance.

**Tech Stack:** Rust, Tauri 2, CPAL/CoreAudio, `webrtc-vad`, `whisper-rs` with macOS Metal, SQLite/rusqlite, Vue 3, Pinia, existing audited `ModelRegistry`. No HTTP client, model downloader, cloud ASR, raw PCM IPC, raw PCM persistence, embeddings, or identity claims.

---

## Product boundary and success criteria

The first usable speech experience is deliberately narrower than full diarization:

1. The user obtains a compatible multilingual whisper.cpp model outside WordCovenant, verifies its SHA-256 from trusted model metadata, and explicitly imports it with model-card and license acknowledgement.
2. The user visibly selects that imported ASR model before recording. Starting a microphone session with no valid selected model fails before capture begins; it does not create synthetic text or silently use a network service.
3. CPAL audio stays in native memory. WebRTC VAD separates silence from speech in 10 ms, 16 kHz mono frames; Whisper produces final Chinese transcript emissions for bounded completed windows.
4. Each final emission is persisted through the existing idempotent SQLite/audit transaction and appears in the current session timeline with capture-clock and wall-clock timestamps.
5. Stopping recording seals the current speech window and gives already admitted work one bounded inference opportunity before the terminal session event. Queue saturation, invalid model data, engine failure, or an interrupted input still become explicit audited inference/capture gaps.
6. The first user-visible labels remain `未归类` or manually assigned anonymous clusters. This milestone must not claim automatic speaker distinction, voiceprint recognition, cross-session matching, or real-person identity.

## Architecture decisions

| Decision | Chosen approach | Reason |
| --- | --- | --- |
| Default ASR | `whisper-rs` / whisper.cpp with a user-imported single file artifact | Fits the existing one-file import, SHA-256, license evidence, 16 kHz input, and local-only model provenance contracts. |
| VAD | WebRTC VAD, fixed 10 ms frames | No runtime model file or download; better speech gate than the existing energy threshold. |
| Model activation | Explicit, session-start selection of a registered ASR model | Importing does not silently enable a model; loading revalidates kind and hash before microphone capture is armed. |
| Runtime ownership | Dispatcher owns PCM + segmentation; separate bounded ASR worker owns the Whisper context | Whisper inference cannot stall the sole ingress consumer or CPAL callback. |
| Timeline update | Event carries only session ID/revision; UI reloads durable timeline | Avoids high-frequency text payloads and preserves the existing audited query boundary. |
| Automatic speaker separation | Later M4: imported ECAPA/WeSpeaker ONNX embedding model plus session-only clustering | Whisper does not diarize. Audio energy, pitch, or text heuristics are not an acceptable substitute. |

Apple Speech.framework is not the default adapter: language resources and model provenance are system-managed, cannot be hash/audit verified, and do not fit the explicit user-import guarantee. It may be evaluated later as a clearly labelled experimental adapter after a separate egress audit.

## Milestones

| Milestone | User-visible result | Exit criteria |
| --- | --- | --- |
| M2.3a: real local ASR | User imports/selects Whisper model, records, sees final timestamped transcript text | Offline capture-to-transcript path passes unit/integration tests and manual macOS test with a real microphone. |
| M2.3b: interaction polish | Current transcript updates during recording; stop reliably completes the last utterance | No stale timeline, misleading engine-ready state, or dropped sealed utterance in stop tests. |
| M3.1: speaker model readiness | User imports a separate embedding model with scope/retention disclosure | Bundle validation, model provenance, and no-embedding-persistence constraints pass. |
| M4: automatic anonymous separation | Final spans receive confidence-aware session-only `说话人 N` assignments, with manual correction retained | Two-speaker/overlap/short-utterance benchmark and false-merge handling are documented; no identity or cross-session lookup exists. |
| M5: quality and operations | Resampling quality, model diagnostics, performance telemetry, release signing/notarization | Device matrix, offline release build, retention controls, and failure recovery are accepted. |

## Task 1: Lock dependencies and model compatibility contract

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/Cargo.lock`
- Modify: `src-tauri/src/inference/model_registry.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Test: `src-tauri/src/inference/model_registry.rs`
- Create: `docs/third-party/whisper-rs.md`
- Create: `docs/third-party/webrtc-vad.md`

**Step 1: Write failing format and selection tests.**

Add tests that reject an ASR artifact whose `input_format` is not accepted by the Whisper adapter, reject a registered VAD/embedding model selected as ASR, and reject a hash-replaced managed artifact before session start.

**Step 2: Add pinned dependencies and declarations.**

Pin `webrtc-vad` and `whisper-rs` to reviewed versions. Enable only the necessary whisper.cpp/Metal build features for macOS. Record version, upstream license, source URL, and required model artifact format in third-party notices. Do not add an HTTP client, a model URL, auto-download code, or a cloud fallback.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::model_registry --lib`

Expected: the accepted format and model-kind paths pass; mismatches fail before capture.

**Step 4: Commit.**

```sh
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/inference/model_registry.rs docs/third-party
git commit -m "build: add local speech inference dependencies"
```

## Task 2: Implement WebRTC VAD as a native-only detector

**Files:**
- Create: `src-tauri/src/inference/webrtc_vad.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Modify: `src-tauri/src/inference/pipeline.rs`
- Test: `src-tauri/src/inference/webrtc_vad.rs`
- Test: `src-tauri/src/inference/pipeline.rs`

**Step 1: Write failing detector tests.**

Cover 16 kHz mono 160-sample frames, deterministic speech/silence fixtures, invalid frame sizes, non-finite samples, and flush behavior. Verify detection state does not cross a capture discontinuity and does not serialize samples.

**Step 2: Implement a buffered `SpeechActivityDetector`.**

Convert finite normalized `f32` samples to saturating signed PCM only inside native memory. Feed exact 10 ms frames to WebRTC VAD with a documented aggressiveness setting. Retain incomplete samples only until the next packet or finish; never log, persist, or return those samples.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::webrtc_vad inference::pipeline --lib`

Expected: bounded segmentation passes and the existing energy detector remains usable for fixtures.

**Step 4: Commit.**

```sh
git add src-tauri/src/inference/webrtc_vad.rs src-tauri/src/inference/mod.rs src-tauri/src/inference/pipeline.rs
git commit -m "feat: add local WebRTC voice detection"
```

## Task 3: Implement verified local Whisper adapter

**Files:**
- Create: `src-tauri/src/inference/whisper_cpp.rs`
- Modify: `src-tauri/src/inference/asr.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Test: `src-tauri/src/inference/whisper_cpp.rs`

**Step 1: Write adapter-boundary tests.**

Test that construction rejects non-ASR metadata, unsupported artifact format, a missing/replaced managed file, invalid sample rate/channel count, and no-speech output. Use a tiny local test model only when licensing permits; otherwise isolate model-independent validation and mark real-model verification as a manual acceptance case.

**Step 2: Build the adapter.**

Create `WhisperCppAsrEngine` from a `RegisteredModel` plus a native verified artifact path. Configure multilingual language selection explicitly for `zh`, disable translation, use bounded token/segment limits, and return only final `TranscriptEmission` values. Map Whisper segment offsets to the request capture range; bound and validate text/word timing before yielding `AsrResponse`. Include the registered model's provider/id/version/SHA in every emission provenance.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::whisper_cpp inference::asr --lib`

Expected: construction and output validation pass without an outbound operation.

**Step 4: Commit.**

```sh
git add src-tauri/src/inference/whisper_cpp.rs src-tauri/src/inference/asr.rs src-tauri/src/inference/mod.rs
git commit -m "feat: add verified local whisper adapter"
```

## Task 4: Select and activate an ASR profile explicitly

**Files:**
- Create: `src-tauri/src/inference/profile.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/state.rs`
- Test: `src-tauri/src/commands.rs`

**Step 1: Write failing activation tests.**

Prove an empty profile, unknown model, wrong kind, unsupported format, and hash mismatch reject `start_session` before `CaptureService::prepare`. Prove a valid profile produces a native ASR/VAD bundle without exposing a filesystem path to Tauri.

**Step 2: Implement a compact serializable profile projection.**

Store model IDs and user-facing display metadata only. Keep actual paths and runtime contexts native. Set the active profile through a deliberate command and expose selection state; importing remains separate from activation.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml state commands --lib`

Expected: an invalid profile never transitions into microphone recording.

**Step 4: Commit.**

```sh
git add src-tauri/src/inference/profile.rs src-tauri/src/state.rs src-tauri/src/commands.rs src-tauri/src/lib.rs
git commit -m "feat: require explicit local ASR profile"
```

## Task 5: Move real inference off the ingress consumer

**Files:**
- Modify: `src-tauri/src/audio/native_runtime.rs`
- Modify: `src-tauri/src/audio/dispatcher.rs`
- Modify: `src-tauri/src/audio/service.rs`
- Test: `src-tauri/src/audio/native_runtime.rs`
- Test: `src-tauri/src/audio/dispatcher.rs`

**Step 1: Write failing concurrency and lifecycle tests.**

Use a blocking fake ASR engine to prove ingress/meter consumption remains live while inference runs. Test fixed job/result capacity, failure conversion to a gap, no second ingress consumer, exact shutdown ownership, and a sealed final window completing during shutdown.

**Step 2: Implement native engine ownership.**

Add `NativeInferenceEngines` and `NativeCaptureRuntime::new_with_engines(...)`, retaining the existing no-engine constructor for regression coverage. Inject WebRTC VAD into the segmenter. Let a single bounded ASR worker own the non-`Sync` Whisper context and deliver outcomes to the existing result queue.

**Step 3: Preserve terminal accounting.**

On stop, stop CPAL production first, finish the segmenter, and allow already admitted jobs the configured bounded completion budget. Record jobs that cannot complete as terminal gaps only after that budget; do not silently discard or block shutdown indefinitely.

**Step 4: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audio::native_runtime audio::dispatcher audio::service --lib`

Expected: real engine path works, missing engine still produces the explicit current gap, and stop retains a final-window path.

**Step 5: Commit.**

```sh
git add src-tauri/src/audio/native_runtime.rs src-tauri/src/audio/dispatcher.rs src-tauri/src/audio/service.rs
git commit -m "feat: run local ASR outside capture ingress"
```

## Task 6: Persist and project live final transcripts

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src/App.vue`
- Modify: `src/stores/session.ts`
- Test: `src-tauri/src/state.rs`
- Test: `src/stores/session.spec.ts`

**Step 1: Write failing projection tests.**

Prove a persisted native final advances a compact revision/counter but sends no PCM. In the store, prove only the active recording session refreshes and repeated projections with unchanged completed-final count do not issue duplicate timeline reads.

**Step 2: Implement compact timeline projection.**

Only after SQLite plus audit success, emit `{ sessionId, revision }`. In the frontend, subscribe to that notification, reject stale/non-active revisions, load through existing `listTimeline(sessionId)`, and merge revisioned spans. On stop, force one final refresh after backend drain completes.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml state --lib && pnpm test --run src/stores/session.spec.ts`

Expected: final transcript lines become visible during a real microphone session and the UI does not poll uncontrolledly.

**Step 4: Commit.**

```sh
git add src-tauri/src/state.rs src-tauri/src/lib.rs src/App.vue src/stores/session.ts src/stores/session.spec.ts
git commit -m "feat: refresh live local transcripts"
```

## Task 7: Add the minimal model-selection UI

**Files:**
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/stores/models.ts`
- Modify: `src/components/ModelRegistryPanel.vue`
- Modify: `src/components/CaptureStatus.vue`
- Modify: `src/assets/main.css`
- Test: `src/components/ModelRegistryPanel.spec.ts`
- Test: `src/stores/models.spec.ts`

**Step 1: Write UI/state tests.**

Cover an empty model state, a selected compatible ASR model, incompatible kinds, verified import metadata, disabled record control without a usable model, and model-load failure messaging. Assert model paths and raw PCM never render.

**Step 2: Implement flat white/gray selection feedback.**

Keep the current restrained flat system: a compact selected-model control, visible local-only status, model compatibility/loading state, and an import action. Do not add cards inside cards, decorative borders, model download links, or network toggles to this workflow.

**Step 3: Run frontend checks and visual QA.**

Run: `pnpm test --run src/components/ModelRegistryPanel.spec.ts src/stores/models.spec.ts && pnpm type-check && pnpm build`

Expected: responsive desktop/mobile controls remain readable and no unselected model can be mistaken for an active engine.

**Step 4: Commit.**

```sh
git add src/types.ts src/lib/wordCovenantApi.ts src/stores/models.ts src/components/ModelRegistryPanel.vue src/components/CaptureStatus.vue src/assets/main.css
git commit -m "feat: select local transcription model"
```

## Task 8: Perform M2.3 offline and hardware acceptance

**Files:**
- Create: `docs/plans/2026-08-10-m2-3-real-local-speech-acceptance.md`
- Modify: `docs/plans/2026-08-07-word-covenant-roadmap.md`
- Modify: `README.md`

**Step 1: Add a real-device acceptance script.**

Include model import/SHA/license acknowledgement, permission allow/deny, 16 and 48 kHz sources, quiet room, Chinese speech, deliberate silence, stop during speech, device interruption, model tamper, and offline network monitor checks. State that only actual mic/model runs demonstrate recognition quality.

**Step 2: Run regression and release checks.**

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

**Step 3: Commit.**

```sh
git add docs/plans README.md
git commit -m "docs: define real local speech acceptance"
```

## Follow-up: M4 automatic anonymous speaker separation

Do not start this feature until M2.3 is accepted with real local ASR. Its plan must define and test the following before implementation:

1. An explicit multi-file or single-artifact embedding-model import format, SHA/license evidence, and a session-level enablement profile.
2. Local-only inference that retains embeddings in bounded native memory. No raw embedding, PCM, or voiceprint profile enters SQLite, the audit payload, logs, WebView, or cross-session storage.
3. Minimum speech duration, uncertainty state, overlap policy, and a bounded session-only clustering algorithm. Assignments must be labelled anonymous and confidence-aware; ambiguity remains unassigned.
4. A way to apply automatic assignments through the existing append-only speaker-correction model, retaining manual rename/reassign as the user override.
5. Evaluation recordings with consent, false-merge/false-split metrics, and an explicit non-goal statement: no named-person recognition, biometric authentication, or cross-session tracking.

