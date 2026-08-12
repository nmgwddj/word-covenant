# Capture Threshold and Device Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let people configure the local speech RMS threshold (default `-10 dBFS`) and make microphone-device selection reliably visible and usable before recording starts.

**Architecture:** Persist one local capture-preference record in the existing SQLite audit store and keep a validated in-memory copy in `AppState`. Snapshot the value immediately before a native microphone session starts, convert dBFS to normalized RMS for that session only, and record it in the capture-start audit payload. Resolve idle device selection from the configured UID and current directory while treating the lifecycle device as authoritative during active capture.

**Tech Stack:** Rust, Tauri 2 commands, SQLite/rusqlite, CPAL/CoreAudio, Vue 3, Pinia, TypeScript, Vitest.

---

### Task 1: Define and persist capture preferences

**Files:**

- Modify: `src-tauri/src/audio/native_runtime.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Test: `src-tauri/src/audio/native_runtime.rs`
- Test: `src-tauri/src/audit/store.rs`

**Step 1: Write failing unit tests**

Add tests proving `SpeechDetectionSettings::default()` is `-10`, accepts the inclusive `-60..=0` dBFS range, rejects values outside it, and converts `-10 dBFS` to an RMS threshold near `0.31622776`.

Add temporary-store tests proving a missing `capture_preferences` row reads as the default and a saved setting survives reopening the SQLite store.

**Step 2: Run focused tests to verify they fail**

Run: `cargo test --manifest-path src-tauri/Cargo.toml speech_detection --lib`

Expected: failures because the settings type and persistence API do not yet exist.

**Step 3: Implement minimal persistence and conversion**

Add a serializable `SpeechDetectionSettings { rms_threshold_dbfs: i8 }`, validation, and a `normalized_rms_threshold()` helper using `10_f32.powf(dbfs as f32 / 20.0)`. Create one-row `capture_preferences` storage in `AuditStore`; invalid rows must never be surfaced as active settings.

**Step 4: Run focused tests**

Run: `cargo test --manifest-path src-tauri/Cargo.toml speech_detection --lib`

Expected: PASS.

### Task 2: Wire the setting through AppState and native capture

**Files:**

- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/audio/service.rs`
- Modify: `src-tauri/src/audio/native_runtime.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/state.rs`
- Test: `src-tauri/src/commands.rs`

**Step 1: Write failing state/command tests**

Cover get/set command serialization, rejection of invalid dBFS values, rejection while a native session is preparing or recording, and a capture-start audit payload containing `speechRmsThresholdDbfs`.

**Step 2: Run focused tests to verify they fail**

Run: `cargo test --manifest-path src-tauri/Cargo.toml speech_detection_settings --lib`

Expected: failures because state methods and commands do not yet exist.

**Step 3: Implement snapshot semantics**

Load settings during `AppState::open`, guard them with a mutex, and expose typed get/set methods. Before the service creates a runtime, copy the settings once and build `NativeCaptureRuntimeConfig` from it. Do not allow the worker configuration to change mid-session. Include the applied value in the existing capture-start audit event.

**Step 4: Run focused tests**

Run: `cargo test --manifest-path src-tauri/Cargo.toml speech_detection_settings --lib`

Expected: PASS.

### Task 3: Make input-device selection resilient and accurately projected

**Files:**

- Modify: `src-tauri/src/audio/service.rs`
- Modify: `src-tauri/src/audio/cpal_input.rs`
- Test: `src-tauri/src/audio/service.rs`
- Test: `src-tauri/src/audio/cpal_input.rs`

**Step 1: Write failing tests**

Add service tests for showing a selected idle device immediately after selection, falling back to the default device after a stale UID disappears, and incrementing the projection revision for visible selection changes. Add CPAL-directory tests for skipping an individual bad device rather than returning an empty directory.

**Step 2: Run focused tests to verify they fail**

Run: `cargo test --manifest-path src-tauri/Cargo.toml input_device --lib`

Expected: failures for the idle selection and directory-resilience cases.

**Step 3: Implement state and directory fixes**

Use the lifecycle device only while it is active; otherwise resolve `selected_device_uid` against the current list. Preserve a last-known-good directory if enumeration fails, remove stale selection UIDs, choose the default when present, and bump projection revision whenever a visible selection changes. Skip only unreadable CPAL devices rather than failing the whole scan. Refresh the directory after microphone permission is checked and before stream opening.

**Step 4: Run focused tests**

Run: `cargo test --manifest-path src-tauri/Cargo.toml input_device --lib`

Expected: PASS.

### Task 4: Build the local recording-settings UI

**Files:**

- Create: `src/components/CaptureSettingsPanel.vue`
- Create: `src/components/CaptureSettingsPanel.spec.ts`
- Create: `src/stores/settings.ts`
- Create: `src/stores/settings.spec.ts`
- Modify: `src/types.ts`
- Modify: `src/lib/wordCovenantApi.ts`
- Modify: `src/App.vue`
- Modify: `src/components/CaptureStatus.vue`
- Modify: `src/components/CaptureStatus.spec.ts`
- Modify: `src/assets/main.css`
- Modify: `components.d.ts`

**Step 1: Write failing Vitest tests**

Cover local default/loading/saving/error behavior in the store; slider, number input, reset control, RMS and peak readout, and locked controls in the panel; selected-device display and non-interactive empty directory state in `CaptureStatus`.

**Step 2: Run focused tests to verify they fail**

Run: `pnpm vitest run src/stores/settings.spec.ts src/components/CaptureSettingsPanel.spec.ts src/components/CaptureStatus.spec.ts`

Expected: failures because the new panel/store and improved empty state do not yet exist.

**Step 3: Implement the flat settings drawer**

Expose get/set settings through Tauri with a browser-preview fallback that only stores state in memory. Add a compact white/gray right-side drawer opened by the existing tune icon. Use a `-60..0` integer slider and numeric input, a reset icon to `-10`, and show both live RMS and peak. Lock device and threshold changes while preparation or recording is active. When no device exists, replace the empty select with a clear static unavailable state and a labeled refresh icon.

**Step 4: Run focused tests**

Run: `pnpm vitest run src/stores/settings.spec.ts src/components/CaptureSettingsPanel.spec.ts src/components/CaptureStatus.spec.ts`

Expected: PASS.

### Task 5: Integrate and verify

**Files:**

- Verify: `src-tauri/src/**/*.rs`
- Verify: `src/**/*.ts`
- Verify: `src/**/*.vue`

**Step 1: Run complete automated suites**

Run: `cargo test --manifest-path src-tauri/Cargo.toml`

Run: `pnpm test`

Run: `pnpm typecheck`

Run: `pnpm build`

**Step 2: Perform a local UI smoke test**

Start the development server, verify the settings drawer at desktop and narrow viewports, choose a device while idle, verify it remains displayed, set/reset the threshold, and confirm that no request leaves the machine.

**Step 3: Review the change set**

Run: `git diff --check` and inspect the staged-independent diff to ensure prior user work remains intact.

**Step 4: Commit only with current authorization**

Do not create or push a commit as part of this task without explicit user approval for the final change set.
