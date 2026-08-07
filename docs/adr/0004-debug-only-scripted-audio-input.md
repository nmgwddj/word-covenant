# ADR-0004: Keep scripted audio input local and debug-only

## Status

Accepted

## Context

Remote development cannot reliably access the Mac's microphone. WordCovenant already has a bounded PCM capture contract, session state, local transcript timeline, and audit trail, but it does not yet have a CoreAudio input stream or local ASR implementation. A browser-only synthetic timeline is useful for layout work, but it does not exercise the Rust-owned session and audit path.

## Decision

Add a deterministic scripted PCM source for debug builds. It emits fixed-size 16 kHz mono packets with sample-clock offsets and has a fixed local transcript cue sheet. Typed Tauri commands start the source and advance it from a development-only UI timer. Each due cue becomes a synthetic `TranscriptSpan` through the existing local state and audit path.

The source never opens a microphone, reads a file, retains audio, sends data over the network, or claims that its text came from ASR. The release UI does not expose the control, and the commands reject calls outside debug builds.

## Consequences

### Positive

- Remote development can exercise start/stop, timestamped timeline updates, speaker labels, and local audit behavior without hardware access.
- Fixture timing and text are deterministic, so tests do not depend on microphone permissions, device availability, or model output.
- The public capture contract is exercised with PCM-shaped data before a native adapter lands.

### Negative

- It does not validate microphone permissions, CoreAudio callbacks, VAD, ASR, diarization, device loss, or real acoustic quality.
- The development UI needs a short polling loop until the production event pipeline exists.

## Alternatives Considered

**Browser-only synthetic timeline:** Rejected for this purpose because it bypasses the Rust session and audit path.

**BlackHole or another virtual audio device:** Deferred until the CoreAudio/CPAL backend exists. It will be useful for hardware-adjacent manual tests later, but cannot exercise the current application end to end.

**A background thread that directly mutates UI state:** Rejected because the WebView must continue to receive typed Rust projections, and an unbounded background worker would add lifecycle complexity before real capture exists.
