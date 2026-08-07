# 开发模拟音频输入 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在没有真实麦克风、没有网络连接的远程开发环境中，让调试版 WordCovenant 用确定性的本地 PCM 包和转写脚本演示一段实时会话。

**Architecture:** Rust 新增一个仅本地的脚本采集源，按 16 kHz、单声道、20 ms 包生成 PCM 和采样偏移。`AppState` 保存该源与待发转写提示，Tauri 命令每次推进固定时长并经现有 `append_transcript` 写入时间线与审计链。Vue 只在开发构建展示一个模拟输入控制，并以短定时器调用推进命令和刷新投影。

**Tech Stack:** Rust、Tauri 2、Serde、Vue 3、Pinia、Vitest、现有 SQLite 审计存储。

---

### Task 1: Define the deterministic scripted source

**Files:**
- Create: `src-tauri/src/audio/demo_source.rs`
- Modify: `src-tauri/src/audio/mod.rs`
- Test: inline Rust tests in `src-tauri/src/audio/demo_source.rs`

**Step 1: Write the failing source tests.**

Verify that a source emits nothing before `start`, then emits sequential 16 kHz mono `CapturePacket` values whose offsets advance by 320 frames. Verify that the cue sheet contains anonymous speaker IDs and deterministic time ranges.

**Step 2: Run the focused test.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audio::demo_source::`

Expected: FAIL because the module does not exist.

**Step 3: Implement the minimal source.**

Generate a short local waveform in memory, with silence between cue ranges. Expose its fixed cue sheet and elapsed sample-clock time. Do not read files, access microphones, create threads, or add dependencies.

**Step 4: Run the focused test.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audio::demo_source::`

Expected: PASS.

### Task 2: Connect the source to sessions and audited transcript spans

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: inline Rust tests in `src-tauri/src/state.rs`

**Step 1: Write failing state tests.**

Start a demo session, advance until the first cue is due, and assert the timeline contains a `synthetic` span with a session-relative timestamp and the audit trail still verifies. Assert that a normal active session prevents a second demo session and that no demo command succeeds in release builds.

**Step 2: Run the focused test.**

Run: `cargo test --manifest-path src-tauri/Cargo.toml state::`

Expected: FAIL until the demo state and commands are present.

**Step 3: Implement the typed projection.**

Store an optional active scripted source in `AppState`. `start_development_mock_session` creates a normal session, and `advance_development_mock` drains a bounded number of 20 ms packets and appends only newly due cue spans. Return a serializable progress projection containing the session ID, packet count, newly emitted spans, and exhaustion state. The Vue store uses the existing `stop_session` path when the script completes.

**Step 4: Register protected Tauri commands.**

Expose typed `start_development_mock_session` and `advance_development_mock` commands only in debug builds. They must never change egress policy.

**Step 5: Run Rust formatting and tests.**

Run: `cargo fmt --manifest-path src-tauri/Cargo.toml --check && cargo test --manifest-path src-tauri/Cargo.toml`

Expected: PASS.

### Task 3: Add a development-only control surface

**Files:**
- Create: `src/components/DevelopmentCaptureControl.vue`
- Create: `src/components/DevelopmentCaptureControl.spec.ts`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/stores/session.ts`
- Modify: `src/App.vue`
- Modify: `src/assets/main.css`

**Step 1: Write the failing component and store tests.**

Verify the control labels the source as development-only and local-only, emits one start request, and disables duplicate starts. Verify the session store updates the active session and merges returned spans in chronological order.

**Step 2: Run focused frontend tests.**

Run: `pnpm test --run src/components/DevelopmentCaptureControl.spec.ts src/stores/session.spec.ts`

Expected: FAIL before the component and store behavior exist.

**Step 3: Implement the adapter and polling lifecycle.**

Add typed API methods for the two commands plus a browser-preview fallback with identical deterministic progress. In development builds only, use a 100 ms timer while simulation is active. Ensure cleanup on unmount and after completion. Never call `fetch`, request microphone permission, or expose the control in production UI.

**Step 4: Implement restrained flat UI.**

Place a compact development marker and one icon-plus-text start command next to recording controls. Use existing white/gray tokens; do not add decorative cards or a prominent warning surface.

**Step 5: Run focused frontend tests and static checks.**

Run: `pnpm test --run src/components/DevelopmentCaptureControl.spec.ts src/stores/session.spec.ts && pnpm type-check`

Expected: PASS.

### Task 4: Verify the complete local demo path

**Files:**
- Modify: `README.md`
- Test: existing Rust and frontend suites

**Step 1: Document the developer-only limitation.**

Add a concise developer note stating that simulated input produces local synthetic transcript cues and is not microphone, ASR, or voiceprint validation.

**Step 2: Run the complete verification matrix.**

Run:

```sh
pnpm test --run
pnpm type-check
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

Expected: all commands pass without opening a microphone or sending network traffic.

**Step 3: Manual debug check.**

Run `pnpm tauri dev`, start a simulated input session, confirm that three timestamped synthetic spans appear over the scripted interval, then verify that the session stops automatically.
