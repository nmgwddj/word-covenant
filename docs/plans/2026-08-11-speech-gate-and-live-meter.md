# Speech Gate and Live Meter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent silence and low-level noise from producing local Whisper transcript hallucinations, while showing a live, privacy-preserving input-level contour during an active recording session.

**Architecture:** The native capture path remains the only PCM consumer. A VAD result is accepted only when its 10 ms frame clears a local RMS threshold, and a sealed utterance is dispatched only after enough accepted speech frames have accumulated. Whisper's per-segment no-speech probability then filters remaining likely-silent output before durable persistence. The WebView uses the existing `CaptureProjection.meter` telemetry only; it never receives, stores, or reconstructs PCM.

**Tech Stack:** Rust, Tauri 2, CPAL/CoreAudio, WebRTC VAD, whisper-rs/whisper.cpp, Vue 3, Vitest. No HTTP client, cloud ASR, raw-audio IPC, or raw-audio persistence.

---

## Acceptance criteria

1. A single VAD false positive on quiet input does not create an ASR request.
2. At least 200 ms of contiguous qualifying speech is required before an utterance can be transcribed; pre-roll and hangover audio remain attached to a qualified utterance but do not count as speech.
3. The VAD continues to receive every frame, including quiet frames, so its native state resets normally.
4. Whisper segments whose reported no-speech probability is at or above the selected threshold are discarded. A fully suppressed result is successful empty output, not an `EngineFailed` inference gap.
5. While native recording is active, the UI displays a compact live audio contour and peak dB value using projection RMS/peak/clipping values only. It is stable when idle or awaiting meter data.
6. No new runtime network operation, audio persistence, or audio payload to the WebView is added.

## Task 1: Add testable local speech admission rules

**Files:**

- Modify: `src-tauri/src/inference/pipeline.rs`
- Modify: `src-tauri/src/inference/webrtc_vad.rs`
- Test: `src-tauri/src/inference/pipeline.rs`
- Test: `src-tauri/src/inference/webrtc_vad.rs`

**Step 1: Write failing segmenter tests.**

Add fixture detectors and exact 10 ms frames proving: quiet VAD-positive frames cannot start a window; 19 qualifying frames cannot dispatch; 20 qualifying frames plus hangover dispatch one request; an under-length active utterance is discarded on `finish`; and normal speech retains pre-roll/hangover audio.

**Step 2: Implement the admission contract.**

Add `minimum_speech_frames` to `SpeechPipelineConfig` with a 20-frame default. Track the longest contiguous run of accepted VAD speech frames in `ActiveUtterance`; have `finalize_active` return no request when that run is below the minimum. Wrap the production WebRTC detector in a local RMS gate of `-50 dBFS` (approximately `0.00316` normalized RMS), but invoke WebRTC VAD before applying that boolean gate on every frame.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline inference::webrtc_vad --lib`

Expected: segment boundaries and VAD fixtures pass, with no outbound activity.

## Task 2: Filter silent Whisper output without manufacturing a failure

**Files:**

- Modify: `src-tauri/src/inference/whisper_cpp.rs`
- Modify: `src-tauri/src/state.rs`
- Test: `src-tauri/src/inference/whisper_cpp.rs`
- Test: `src-tauri/src/state.rs`

**Step 1: Write failing adapter/state tests.**

Extend the adapter fixture with `no_speech_probability`. Assert a segment at or above `0.60` is omitted, a normal segment remains, and an empty valid response is acknowledged without inserting an inference gap.

**Step 2: Implement segment filtering.**

Read `WhisperSegment::no_speech_probability()` and token log probabilities when collecting native segments. Route the adapter through a small pure filter so test fixtures do not require a model. Keep high-confidence speech even when its no-speech value is elevated. Mark an empty Whisper result explicitly as `NoSpeech`; in `persist_native_outcome`, only that declared outcome commits without timeline or gap writes.

**Step 3: Run focused tests.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml inference::whisper_cpp state --lib`

Expected: silent output produces neither transcript nor `EngineFailed` gap; speech output preserves timestamp mapping and model provenance.

## Task 3: Surface the existing meter as a compact real-time contour

**Files:**

- Create: `src/components/LiveAudioMeter.vue`
- Create: `src/components/LiveAudioMeter.spec.ts`
- Modify: `src/App.vue`
- Modify: `components.d.ts`
- Test: `src/components/LiveAudioMeter.spec.ts`

**Step 1: Test component behavior.**

Cover idle, awaiting first projection, normal signal, clipping, and malformed dBFS values. Confirm that visual state is derived only from `CaptureMeter` scalars.

**Step 2: Integrate the component.**

Place the meter beside the real recording status/device information. Feed it the latest `capture-projection` meter and show it only for the active native microphone lifecycle. Keep the UI flat, white/gray, compact, and free of new bordered panels; use the existing recording color only for clipping.

**Step 3: Run focused frontend tests.**

Run: `pnpm vitest run src/components/LiveAudioMeter.spec.ts`

Expected: deterministic bars, peak readout, and accessible labels pass without an audio payload.

## Task 4: Verify the combined behavior

**Files:**

- Test: native Rust unit suites and frontend unit/type/build suites

**Step 1: Run regression suites.**

Run:

```sh
cargo test --manifest-path src-tauri/Cargo.toml --lib
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

**Step 2: Perform macOS manual acceptance.**

Build the Apple Silicon app, start a real microphone session, verify a quiet room leaves the timeline unchanged while the meter remains near its noise floor, then speak a normal Chinese phrase for more than 200 ms and verify one timestamped local transcript appears. Disconnect or select a loopback device and verify any low-level input is visually represented but does not create a transcript.

**Step 3: Commit.**

```sh
git add docs/plans/2026-08-11-speech-gate-and-live-meter.md src-tauri/src/inference src-tauri/src/state.rs src/App.vue src/components/LiveAudioMeter.vue src/components/LiveAudioMeter.spec.ts components.d.ts
git commit -m "feat: suppress silent transcription and show input meter"
```
