# M2.2 Native Inference Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect macOS CPAL ingress to a local, bounded speech-inference bridge with a single PCM consumer, explicit terminal gaps, safe lifecycle ordering, and no network or PCM IPC.

**Architecture:** A `CaptureDispatcher` becomes the unique reader of `CaptureIngress`. It projects meter values and passes bounded speech windows to a single ASR worker through fixed-capacity queues. The state layer claims and durably commits outcomes, preserving a final transcript or a distinct `InferenceGap` for every completed window. CPAL is prepared before recording is published and shutdown drains or terminally accounts for work before the session stop event.

**Tech Stack:** Rust, Tauri 2, CPAL/CoreAudio on macOS, existing `crossbeam_queue::ArrayQueue`, SQLite/rusqlite, existing audit hash chain, existing local inference traits and deterministic fixtures. No new dependency, HTTP client, model download, WebView PCM transport, or outbound permission.

---

## Scope and non-goals

M2.2 establishes native ingress, queue, lifecycle, and evidence semantics. It does not add a production VAD, whisper.cpp, Metal, a 44.1 kHz resampler, speaker diarization, cloud inference, or a hardware-quality claim. A missing executable local ASR adapter is a durable `local_engine_unavailable` inference gap, never fixture text. The implementation must continue to work entirely offline even when egress is disabled or absent.

`CaptureGap` remains a physical capture discontinuity. `InferenceGap` means a known captured range reached the native bridge but has a terminal non-transcript outcome. Neither type stores PCM in SQLite, audit events, logs, or Tauri commands.

## Non-functional requirements

- The CPAL callback performs finite normalization, source-offset accounting, and one bounded ingress write only. It never locks `AppState`, runs inference, persists data, emits Tauri events, or uses network APIs.
- Every queue and temporary held result has a fixed maximum. A full queue produces a range-bearing terminal outcome rather than an allocation or an unreported discard.
- Final ASR output, inference gaps, and stop events use the segment's continuous-time capture clock and matching wall-clock mapping.
- Results carry a session/runtime generation and capture-segment fence; stale results cannot enter a restarted session or post-stop timeline.
- SQLite writes bind the domain record and its audit event atomically. A failed persistence attempt can be retried without losing or duplicating the outcome.

## Task 1: Add durable inference-gap and capture-event bindings

**Files:**
- Create: `src-tauri/src/inference/gap.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Modify: `src-tauri/src/audit/hash_chain.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Test: `src-tauri/src/inference/gap.rs`
- Test: `src-tauri/src/audit/store.rs`

**Step 1: Write failing domain and SQLite tests.**

Create an `InferenceGap` fixture with a stable UUID, session id, dispatcher generation, capture-segment id, optional job id, start/end `CapturePoint`, stage and reason. Test invalid ranges, empty identity, and unrecognised enum values. Add store tests proving that append/reopen preserves a gap and that tampering, deletion, duplicate binding, or a mismatched audit payload makes `AuditStore::verify()` false.

**Step 2: Run focused tests.**

```sh
cargo test --manifest-path src-tauri/Cargo.toml inference::gap audit::store --lib
```

Expected: FAIL because neither the gap type nor its audited persistence exists.

**Step 3: Add minimal domain and audited storage.**

Define `InferenceGapStage` and `InferenceGapReason` separately from `CaptureGapReason`. Add `AuditKind::InferenceGapRecorded`. Add `inference_gaps` with `id`, session/runtime/segment/job references, range, stage/reason, and `audit_event_id`; make the record immutable and the binding unique. Add `append_inference_gap_with_audit`, `list_inference_gaps`, and verification that checks audit-event kind, run id, payload hash, and a one-to-one immutable row. Keep existing capture segment/gap storage schema-compatible in this milestone; do not introduce an irreversible backfill merely to tighten historical record links. Do not persist samples or transcript content in a gap.

**Step 4: Run focused tests.**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/inference/gap.rs src-tauri/src/inference/mod.rs src-tauri/src/audit/hash_chain.rs src-tauri/src/audit/store.rs
git commit -m "feat: audit bounded inference gaps"
```

## Task 2: Separate speech segmentation from synchronous ASR

**Files:**
- Modify: `src-tauri/src/inference/pipeline.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Test: `src-tauri/src/inference/pipeline.rs`

**Step 1: Write failing segmenter tests.**

Extract the current packet conversion, discontinuity detection, pre-roll, hangover, maximum-window and finish cases into tests that yield owned `AsrRequest` values without calling an engine. Test that a discontinuity seals the preceding window and does not join it to the next range. Preserve the existing M2.1 fixture tests through a compatibility wrapper.

**Step 2: Run focused tests.**

```sh
cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib
```

Expected: FAIL until the segmenter has a request-producing API.

**Step 3: Implement the smallest compatible split.**

Introduce `SpeechSegmenter<D>` and `SpeechWindowEvent::{Request, Discontinuity}`. It owns the bounded native PCM conversion and VAD state, but not an `AsrEngine`. Keep `SpeechPipeline<D, A>` as the M2.1 wrapper that feeds the segmenter output to its injected synchronous fixture ASR. Ensure all request audio is 16 kHz mono, bounded to the existing 30-second maximum, and derives both endpoints from `CaptureClock`.

**Step 4: Run focused tests.**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/inference/pipeline.rs src-tauri/src/inference/mod.rs
git commit -m "refactor: separate bounded speech segmentation"
```

## Task 3: Implement the native dispatcher, jobs, results, and backpressure

**Files:**
- Create: `src-tauri/src/audio/dispatcher.rs`
- Modify: `src-tauri/src/audio/mod.rs`
- Modify: `src-tauri/src/audio/capture.rs`
- Test: `src-tauri/src/audio/dispatcher.rs`

**Step 1: Write failing deterministic bridge tests.**

Use a fake `CaptureIngress`, deterministic detector, and fixture/unavailable engine to prove that exactly one dispatcher consumer both updates meter values and creates a job. Test zero capacities are rejected; job saturation yields one `job_queue_saturated` gap with the exact capture range and does not invoke ASR; result saturation yields one `result_queue_saturated` gap before ASR is called; an engine failure and unavailable engine become distinct gaps. Test FIFO order, high-water counters, begin/commit/abort delivery, and that an aborted result remains available for retry.

**Step 2: Run focused tests.**

```sh
cargo test --manifest-path src-tauri/Cargo.toml audio::dispatcher --lib
```

Expected: FAIL because no dispatcher contract exists.

**Step 3: Build the bounded bridge.**

Use `ArrayQueue` for jobs and outcomes. Define `AsrJob`, `AsrOutcome`, `DispatcherRuntimeId`, `AsrBridgeConfig`, and compact `AsrQueueMetrics`. The dispatcher owns the sole `try_consume` loop, invokes the segmenter, and uses non-blocking job admission. One native worker processes at most one job at a time and retains at most one pending outcome while the result queue is full. Explicit close/drain methods must guarantee that every admitted request is delivered as a completed response or an inference gap. Projections expose counts and status only; no PCM or transcript text crosses this module's public UI-facing boundary.

**Step 4: Run focused tests.**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/audio/dispatcher.rs src-tauri/src/audio/mod.rs src-tauri/src/audio/capture.rs
git commit -m "feat: add bounded native inference dispatcher"
```

## Task 4: Replace the CPAL meter worker with prepared native runtime

**Files:**
- Modify: `src-tauri/src/audio/cpal_input.rs`
- Modify: `src-tauri/src/audio/service.rs`
- Modify: `src-tauri/src/audio/lifecycle.rs`
- Test: `src-tauri/src/audio/cpal_input.rs`
- Test: `src-tauri/src/audio/service.rs`

**Step 1: Write lifecycle tests before changing macOS glue.**

Test prepare success, unsupported 44.1 kHz preflight rejection, dispatcher startup failure, stream-play failure, and activation/commit failure. In every failure case assert no `Recording` projection, no live stream/worker, no session/timeline, and no fake gap. Test stop and device failure ordering with a fake bridge: producer stops, ingress drains, segmenter finishes, outcomes are available, then the runtime becomes non-recording.

**Step 2: Run focused tests.**

```sh
cargo test --manifest-path src-tauri/Cargo.toml audio::service audio::cpal_input --lib
```

Expected: FAIL because `CpalInput::start` plays immediately and spawns a meter-only consumer.

**Step 3: Implement prepare, activate, arm, and drain.**

Split CPAL construction from `stream.play()`. Build the ingress, parked dispatcher, job/result queues, and paused stream first. Preflight only the currently supported 16 kHz and 48 kHz input formats. On successful play, return the capture anchor; arm the dispatcher only after the caller commits the audit/session start. Remove `spawn_pcm_worker`; meter atomics are updated by the dispatcher. Stop must release the stream before joining workers, and must not join while holding `CaptureService`'s mutex. Preserve callback lock-free behavior and record exact callback-drop ranges where they are known.

**Step 4: Run focused tests.**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/audio/cpal_input.rs src-tauri/src/audio/service.rs src-tauri/src/audio/lifecycle.rs
git commit -m "feat: prepare native capture before recording"
```

## Task 5: Fence and persist native outcomes through AppState

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src-tauri/src/domain/session.rs`
- Test: `src-tauri/src/state.rs`
- Test: `src-tauri/src/commands.rs`

**Step 1: Write failing state and stop-race tests.**

Create a prepared runtime fixture with a session id, dispatcher generation, and capture segment. Prove that a valid final uses the existing durable idempotency transaction, while partial output is never persisted. Prove that a gap gets an `InferenceGapRecorded` event, a store write failure leaves the outcome claim retryable, and a generation mismatch cannot alter a new session. Cover outcomes pending at ingress, job, worker-held, and result stages during stop: each becomes a final or a durable gap before `SessionStopped`.

**Step 2: Run focused tests.**

```sh
cargo test --manifest-path src-tauri/Cargo.toml state commands --lib
```

Expected: FAIL because native ASR output has no runtime/segment fence and capture gaps are destructively drained before persistence succeeds.

**Step 3: Integrate transactionally.**

Add a private native-outcome pump called from the existing macOS projection loop and synchronously from stop. It claims one outcome outside the capture-service lock, validates the active or closing runtime lease, and either persists its final through `append_local_asr_response` or persists an `InferenceGap` in the same audit transaction. It commits only after SQLite succeeds; otherwise it aborts the claim. Derive transcript wall-clock endpoints from the associated capture segment, not the first session anchor. Fence the generation before producer shutdown, reject stale results, clear the mapper only after all final/gap outcomes are terminal, and write `SessionStopped` last. Keep commands and frontend payloads limited to compact status/counters.

**Step 4: Run focused tests.**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/state.rs src-tauri/src/commands.rs src-tauri/src/lib.rs src-tauri/src/domain/session.rs
git commit -m "feat: fence and audit native inference outcomes"
```

## Task 6: Verify, document, and manually exercise macOS behavior

**Files:**
- Modify: `docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md`
- Modify: `docs/plans/2026-08-07-word-covenant-roadmap.md`
- Test: existing Rust and frontend suites

**Step 1: Add M2.2 manual cases.**

Document the exact build, selected 16/48 kHz device requirement, permission deny/allow, startup failure, device interruption, queue saturation, stop while work is pending, missing local engine, restart/generation rejection, and a network-monitor observation. Mark all hardware-only steps pending until run on the same release build; do not use fixture tests as evidence for hardware or model quality.

**Step 2: Run the offline verification matrix.**

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

Expected: all commands pass without downloading a model, adding an HTTP dependency, or sending PCM over Tauri IPC.

**Step 3: Commit.**

```sh
git add docs/plans/2026-08-10-m2-2-native-inference-bridge.md docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md docs/plans/2026-08-07-word-covenant-roadmap.md
git commit -m "docs: define native inference bridge acceptance"
```
