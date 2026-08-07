# WordCovenant

> 凡口头所言，皆立为契约，有据可查，事事落单。

WordCovenant is a macOS-first, local-first desktop workspace for visible conversation capture, timestamped transcript records, anonymous speaker clusters, and user-triggered agent actions.

## Privacy Contract

- Audio, transcript, speaker data, and context stay on the device by default.
- Network egress is denied unless the user visibly enables the session-only outbound switch and separately approves a named tool, its HTTPS origin, and its data scope.
- A transcript, model, skill, hook, or external response cannot grant permissions by itself.
- Agent plans are typed data. They are not shell commands and cannot bypass Rust policy checks.
- The current build does not automatically identify people from voiceprints and does not claim legal non-repudiation.

## Current M0 Slice

- WordCovenant macOS application branding, a restricted Tauri capability set, and a CSP with no wildcard network origin.
- A local recording workspace with an always-visible local-only/recording state and chronological transcript timeline.
- Rust domain contracts for capture sessions, revisioned transcript spans, anonymous speaker clusters, plans, and tool calls.
- Default-deny HTTPS egress policy with a session-only visible master gate plus exact-origin, scope, expiry, and revocation checks.
- Append-only SHA-256 audit chain persisted locally through SQLite.
- A bounded audio capture contract, sample-clock time conversion, explicit gap types, and deterministic test capture source.
- Typed Tauri commands for session lifecycle, timeline queries, egress approval/revocation, local action proposals, and network policy preflight.

The M0 UI uses local synthetic data outside Tauri so it can be inspected in a browser. It labels that data `synthetic`; it is not a microphone or ASR fallback.

## Planned Milestones

The M0 implementation plan is at [docs/plans/2026-08-07-word-covenant-local-first-mvp.md](docs/plans/2026-08-07-word-covenant-local-first-mvp.md). The product roadmap, milestones, acceptance gates, risks, and architecture decisions are at [docs/plans/2026-08-07-word-covenant-roadmap.md](docs/plans/2026-08-07-word-covenant-roadmap.md).

1. M0.1 adds a visible session-only egress master gate. An approved profile remains blocked until the user confirms that gate; it resets to off after restart.
2. M1 adds CoreAudio capture, real session clocks, and visible meter/gap events.
3. M2 adds explicitly imported local VAD and ASR adapters. Models will never auto-download.
4. M3 adds anonymous speaker clustering and user-controlled rename/merge corrections.
5. M4 adds user-triggered Agent planning, local TTS, approval UI, and named outbound HTTP profiles.
6. M5 adds declarative skills, then signed constrained WASM hooks.
7. M6 adds Keychain-backed encryption, retention controls, exports, notarization, and release operations.

## Development

Prerequisites: a current Rust toolchain, pnpm, Xcode command-line tools, and the normal Tauri macOS prerequisites.

```sh
pnpm install
pnpm test --run
pnpm type-check
pnpm check
pnpm tauri dev
```

Run the browser-only preview at `http://127.0.0.1:1420`:

```sh
pnpm vite:dev
```

## Verification

```sh
pnpm test --run
pnpm type-check
pnpm build
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

The test suite covers default-deny egress, approval matching, expiration, revocation, transcript timestamp boundaries, audit-chain tamper detection, local capture queue behavior, and the visible local-only UI states.
