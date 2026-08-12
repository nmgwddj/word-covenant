# Local ASR Quality And Turbo Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve local Mandarin-English transcription quality without adding runtime network access, then replace the bundled Base model with the reviewed `large-v3-turbo-q5_0` artifact.

**Architecture:** Keep PCM and inference entirely native. Replace fixture-grade decimation with a stateful anti-aliasing resampler, make speech energy gating adaptive by default with an explicit manual override, and use a stronger bounded Whisper decoding profile with independent VAD utterances. Preserve raw Whisper text in SQLite while projecting deterministic Simplified Chinese through a versioned local normalizer.

**Tech Stack:** Rust, Tauri 2, CPAL/CoreAudio, `rubato`, WebRTC VAD, `whisper-rs`/whisper.cpp Metal, pure-Rust `ferrous-opencc`, SQLite, Vue 3, Pinia, Node model-staging scripts.

---

### Task 1: Replace 48 kHz sample dropping with anti-aliasing resampling

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/src/inference/pipeline.rs`
- Test: `src-tauri/src/inference/pipeline.rs`
- Modify: `docs/third-party/rubato.md`

**Steps:**

1. Add failing tests that feed an above-Nyquist 48 kHz tone and verify it is attenuated after conversion to 16 kHz, while a speech-band tone retains bounded amplitude and exact capture duration.
2. Pin the reviewed stable `rubato` release and document its license/source.
3. Add a mono 48 kHz to 16 kHz streaming resampler owned by `SpeechSegmenter`; allocate buffers outside the CPAL callback and preserve source-offset discontinuity semantics.
4. Replace `source_offset % 3 == 0` sample dropping with the resampler output and reset its state after ingress discontinuities.
5. Run `cargo test --manifest-path src-tauri/Cargo.toml inference::pipeline --lib` and `cargo fmt --check`.

### Task 2: Make speech energy gating adaptive by default

**Files:**
- Modify: `src-tauri/src/audio/native_runtime.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/stores/settings.ts`
- Modify: `src/components/CaptureSettingsPanel.vue`
- Test: corresponding Rust, Pinia, API, and component test modules

**Steps:**

1. Extend speech settings with `mode: adaptive | manual`; preserve the manual threshold as a stored fallback. This WIP build intentionally recreates local data instead of carrying a schema migration.
2. Add an adaptive noise estimator that updates only on WebRTC-VAD negative frames, applies a 12 dB margin, and clamps the effective RMS threshold to `-42..-24 dBFS`.
3. Keep manual mode as the exact configured `-42..0 dBFS` gate. Reset the estimator after discontinuities. The lower bound is also enforced immediately before Whisper after the reviewed turbo model produced confident subtitle hallucinations for digital silence and a `-43 dBFS` hum fixture.
4. Persist and audit the configured mode and manual threshold. Lock both during capture preparation and recording.
5. Add an Auto/Manual segmented control to the existing recording settings panel; disable the slider while Auto is selected and explain the locally computed behavior without adding network calls.
6. Run focused Rust and Vitest suites, then type-check the frontend.

### Task 3: Improve Whisper decoding for Mandarin-English speech

**Files:**
- Modify: `src-tauri/src/inference/whisper_cpp.rs`
- Test: `src-tauri/src/inference/whisper_cpp.rs`

**Steps:**

1. Add parameter-construction tests for beam search, explicit Chinese-primary language, independent context, and single-segment VAD utterances.
2. Replace greedy `best_of=1` with beam search width 5.
3. Decode each VAD utterance without prior transcript history or a generic instruction prompt; the observed rolling-history design amplified a common subtitle hallucination across independent audio windows.
4. Keep `translate=false`, Chinese-primary decoding, silence suppression, bounded tokens, and all existing path/text privacy constraints.
5. Run `cargo test --manifest-path src-tauri/Cargo.toml inference::whisper_cpp --lib`.

### Task 4: Project deterministic Simplified Chinese without overwriting evidence

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Create: `src-tauri/src/inference/text_normalization.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/domain/transcript.rs`
- Modify: `src/types.ts`
- Test: normalization and projection tests
- Create: `docs/third-party/ferrous-opencc.md`

**Steps:**

1. Pin reviewed pure-Rust `ferrous-opencc` 0.4.0 with only embedded `t2s` dictionaries, and document its Apache-2.0 license/source.
2. Implement a process-local `t2s` normalizer with a fixed profile identifier; English, numbers, and punctuation must remain unchanged.
3. Keep `TranscriptRevision.text` as the raw Whisper value. Normalize only when producing `TranscriptSpan`, and include optional `originalText` plus a normalization profile only when text changes.
4. Test common Traditional-to-Simplified phrases, mixed English terms, replay/idempotency stability, and unchanged raw durable revisions.
5. Run focused domain/state tests.

### Task 5: Upgrade the bundled model supply chain

**Files:**
- Replace: `models/whisper-base.lock.json` with a turbo-specific lock
- Modify: `scripts/stage-bundled-model.mjs`
- Modify: `scripts/stage-bundled-model.test.mjs`
- Modify: `src-tauri/src/inference/bundled_model.rs`
- Modify: `src-tauri/resources/models/manifest.json`
- Replace: `src-tauri/resources/models/ggml-base.bin`
- Update: `src-tauri/resources/third-party/*whisper*`
- Modify: `src-tauri/tauri.conf.json`

**Steps:**

1. Generalize the staging script and native lock validation from the literal Base variant to the reviewed `large-v3-turbo-q5_0` identity while keeping exact filename, revision, size, and SHA-256 checks.
2. Download only the pinned Hugging Face revision to a local build cache, calculate SHA-256, and compare the byte count to the official repository metadata.
3. Assign a new model UUID/version and update the lock, packaged manifest, model card, license notice, Tauri resource mapping, and tests together.
4. Stage the verified model atomically and run `pnpm test:model-stage` plus `pnpm model:verify`.
5. Confirm the application contains no model downloader or HTTP client; build-time acquisition must not change runtime egress policy.

### Task 6: Integrate, benchmark, and package

**Files:**
- Verify: all changed Rust, TypeScript, Vue, script, manifest, and resource files

**Steps:**

1. Run `cargo test --manifest-path src-tauri/Cargo.toml`, `cargo fmt --check`, and `cargo clippy --all-targets -- -D warnings` where platform dependencies permit.
2. Run `pnpm vitest run`, `pnpm type-check`, `pnpm build`, `pnpm test:model-stage`, and `pnpm model:verify`.
3. Build the macOS release bundle and verify the packaged manifest/model SHA-256 without launching any network behavior.
4. Smoke-test local microphone capture, adaptive/manual settings, Simplified-Chinese projection, stop/drain behavior, and the bottom status bar.
5. Report measured model load time, peak memory where available, and utterance latency. Do not replace `/Applications/WordCovenant.app` without separate explicit authorization.
