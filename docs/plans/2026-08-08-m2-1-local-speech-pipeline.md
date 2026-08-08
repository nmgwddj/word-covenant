# M2.1 Local Speech Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route deterministic local PCM through a bounded, native-only 16 kHz speech pipeline so fixture ASR finals reach the existing audited transcript store without persisting partial text.

**Architecture:** Add a Rust-only pipeline below the inference boundary. It accepts borrowed `CapturePacket` data, supports explicit 16 kHz identity and 48 kHz-to-16 kHz conversion, detects source-clock discontinuities, and uses bounded pre-roll/hangover segmentation before calling injected local VAD/ASR fixtures. Development mock input uses this pipeline and hands only final `AsrResponse` values to `AppState`; the real CPAL ingress remains unchanged until M2.2 replaces its single meter worker with a dispatcher.

**Tech Stack:** Rust, existing `CaptureClock`/`CapturePacket`, local inference traits, deterministic fixtures, `rusqlite` audit store, Cargo unit tests. No new package, model download, HTTP client, WebView PCM IPC, or outbound permission.

---

## Scope and Non-goals

M2.1 is deliberately a deterministic contract slice. It proves packet timing,
format conversion, bounded speech segmentation, partial/final handling, and
the local audit path. It is not a real microphone-to-ASR bridge, a production
quality resampler, an actual VAD model, a `whisper.cpp` binding, or a benchmark
claim. The temporary energy/frame fixture is clearly internal test machinery,
not a claim of production voice activity detection.

`CaptureIngress` has one consumer today. M2.1 must not add another consumer:
two `try_consume` loops would race and cause meter and ASR to receive different
PCM packets. M2.2 will replace the consumer with one dispatcher after the
native lifecycle can be manually tested.

### Task 1: Define a bounded native PCM-to-window pipeline

**Files:**
- Create: `src-tauri/src/inference/pipeline.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Test: `src-tauri/src/inference/pipeline.rs`

**Step 1: Write failing conversion and validation tests.**

Add tests that feed 16 kHz mono, 48 kHz mono, and 48 kHz stereo packets into a
`SpeechPipeline`. Assert that output windows are 16 kHz/mono, have finite
samples, retain source-clock endpoints, and downmix stereo frames before
decimation. Add rejected-input tests for an unsupported source rate, malformed
interleaving, non-finite samples, and packet frame-offset discontinuity.

**Step 2: Run the focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib`

Expected: FAIL because the native pipeline does not exist.

**Step 3: Implement the smallest native-only API.**

Introduce a borrowed `NativePcmPacket<'a>` plus `SpeechPipeline<V, A>`. Permit
only 16 kHz identity and deterministic 48 kHz-to-16 kHz conversion in this
batch. The source `starting_sample_offset` remains the time authority; use
`CaptureClock` to derive every output range. Reject rather than silently
approximate unsupported device formats. Do not serialize PCM or retain an
unbounded input buffer.

**Step 4: Run the focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib`

Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/inference/pipeline.rs src-tauri/src/inference/mod.rs
git commit -m "feat: add bounded local speech pipeline"
```

### Task 2: Segment local speech with bounded pre-roll and hangover

**Files:**
- Modify: `src-tauri/src/inference/pipeline.rs`
- Test: `src-tauri/src/inference/pipeline.rs`

**Step 1: Write failing segmentation tests.**

Inject deterministic frame activity into the pipeline and prove that speech
starts with configured pre-roll, includes configured trailing hangover, flushes
at end of stream, and never creates a window beyond the existing 30-second /
480,000-sample inference maximum. A source-clock discontinuity must reset the
active utterance and emit a discontinuity event instead of joining audio on
either side.

**Step 2: Run the focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib`

Expected: FAIL until segmentation state is implemented.

**Step 3: Implement bounded segment assembly.**

Use fixed 10 ms frames at 16 kHz, bounded pre-roll storage, finite hangover,
and an explicit `finish` operation. VAD and ASR adapters execute through their
existing local traits; emitted `AsrResponse` values have no database or Tauri
reference. A partial remains an ASR response only, never pipeline persistence.

**Step 4: Run the focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib`

Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/inference/pipeline.rs
git commit -m "feat: segment local speech with bounded timing"
```

### Task 3: Make fixture ASR safe for multiple utterances

**Files:**
- Modify: `src-tauri/src/inference/mock.rs`
- Test: `src-tauri/src/inference/mock.rs`

**Step 1: Write a failing fixture replay test.**

Call the fixture on two separate windows and assert that their final emissions
carry distinct utterance keys while each response retains its stable
partial-to-final revision sequence.

**Step 2: Run the focused test.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::mock --lib`

Expected: FAIL because the current fixture uses one fixed utterance key.

**Step 3: Implement request-specific deterministic keys.**

Derive the fixture utterance key from the native capture range, not wall-clock
or random state. Preserve replay determinism and existing model provenance.

**Step 4: Run the focused test.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::mock --lib`

Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/inference/mock.rs
git commit -m "test: distinguish fixture ASR utterances"
```

### Task 4: Route the development mock through final ASR persistence

**Files:**
- Modify: `src-tauri/src/audio/development_mock.rs`
- Modify: `src-tauri/src/state.rs`
- Test: `src-tauri/src/audio/development_mock.rs`
- Test: `src-tauri/src/state.rs`

**Step 1: Write failing end-to-end mock tests.**

Advance the PCM mock through speech and assert it returns a final projection
only after pipeline finalization. Verify partial text produces no transcript
revision or FTS entry. Assert one final is persisted through
`append_local_asr_response`, has the source session's capture timestamps, and
leaves the audit chain valid. Replaying the same final must not duplicate it.

**Step 2: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml development_mock --lib`

Expected: FAIL because the mock directly constructs `TranscriptSpan` values.

**Step 3: Replace direct cues with pipeline input.**

Have `DevelopmentMockRunner` feed its packets to a fixture pipeline. Keep the
existing debug-only public progress shape where practical, but expose only
compact final projections. In `AppState`, append output using
`append_local_asr_response`; do not use the legacy direct `append_transcript`
path for mock inference.

**Step 4: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml development_mock --lib`

Expected: PASS.

**Step 5: Commit.**

```sh
git add src-tauri/src/audio/development_mock.rs src-tauri/src/state.rs
git commit -m "feat: route mock audio through local ASR pipeline"
```

### Task 5: Verify the local-only boundary and document the next bridge

**Files:**
- Modify: `docs/plans/2026-08-07-word-covenant-roadmap.md`
- Modify: `docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md`

**Step 1: State the tested boundary.**

Record that M2.1 covers pure Rust pipeline and mock input only. List M2.2 as
the single-ingress-dispatcher work: two-phase capture startup, bounded ASR job
and result queues, explicit inference backpressure/gaps, and a real macOS
manual acceptance run. Do not mark real native ASR or real VAD complete.

**Step 2: Run the complete offline verification matrix.**

Run:

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

Expected: all commands pass without adding a dependency that needs network
access.

**Step 3: Commit.**

```sh
git add docs/plans/2026-08-08-m2-1-local-speech-pipeline.md docs/plans/2026-08-07-word-covenant-roadmap.md docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md
git commit -m "docs: define local speech pipeline acceptance"
```
