# ADR-0005: Use a single native dispatcher and bounded inference outcomes

## Status

Accepted

## Context

M2.1 established a bounded, local Rust speech-pipeline contract for fixture
input. The macOS CPAL path currently has a separate meter-only worker that is
the sole consumer of `CaptureIngress`; it cannot also feed ASR without two
consumers racing for PCM packets. The real capture path must connect to local
inference without blocking the CoreAudio callback, serializing PCM through
Tauri, creating unbounded memory, or making a network request.

Backpressure has an evidentiary consequence. A `CaptureGap` means the source
audio was not captured. It must not be reused for audio that was captured but
could not be inferred or durably projected. The latter needs its own stable,
range-bearing and auditable terminal record. There is no production ASR model
adapter in this milestone, so the bridge must not manufacture transcript text
when a user has only imported model metadata or no executable engine.

## Decision

Create one native `CaptureDispatcher` for each active capture runtime. It is
the only code that calls `CaptureIngress::try_consume`. For each borrowed
packet it calculates compact meter telemetry and drives a bounded segmenter;
the callback continues to do only sample normalization and a lock-free ingress
write. The dispatcher copies only completed, bounded 16 kHz ASR requests into
a fixed-capacity job queue. One worker processes jobs sequentially, and a
fixed-capacity result queue delivers an owned `AsrOutcome` to the state layer.

Both queues use existing local `ArrayQueue` primitives. Admission is
non-blocking. Queue saturation, missing local engine, engine failure, stop
before inference, and persistence failure become explicit `InferenceGap` or
retryable outcome states. A result is claimed with `begin`, removed only by
`commit`, and retained by `abort`; it is never silently dropped. The maximum
additional held outcome is one worker-owned item, so memory remains bounded.

`InferenceGap` is distinct from `CaptureGap` and carries a stable id, session,
dispatcher generation, capture segment, job id where available, capture and
wall-clock range, stage, and reason. It is persisted with an
`InferenceGapRecorded` audit event in the same SQLite transaction. New
inference-gap rows have durable, one-to-one audit-event links; verification
checks both directions, including payload digests and immutable bindings.
Existing M1 capture rows retain their compatible audit-chain validation and
are not subject to an irreversible schema backfill in this milestone.

Startup has three private stages: prepare permission/device/config, construct
the parked ingress and dispatcher resources, then play the CPAL stream. Only
after the stream handoff and initial audit transaction succeed is the runtime
armed and `Recording` published. A failed stage tears down workers and the
stream without a visible session or transcript. Shutdown fences the runtime
generation, stops CPAL, drains ingress, seals jobs, produces a final or gap for
every admitted job, commits/acknowledges outcomes, clears transient ASR state,
and only then records `SessionStopped`. Results from a stale generation are
rejected before they can change the timeline.

All audio, requests, outcomes, model execution, and persistence remain inside
the native process. UI projections expose only compact runtime state and
counts. This decision adds no HTTP client, model download, egress permission,
or PCM Tauri command.

## Consequences

### Positive

- Metering and inference see the same capture sequence through one ingress
  consumer.
- Queue pressure and unavailable local inference produce visible, verifiable
  evidence instead of ambiguous missing transcript text.
- The CoreAudio callback has bounded work and no application locks, SQLite,
  IPC, or outbound path.
- A real model adapter can later replace the explicit unavailable-engine
  outcome without changing lifecycle or audit semantics.

### Negative

- The first bridge has more native lifecycle states and a stricter stop path.
- A single ASR worker favors deterministic order and bounded resources over
  parallel throughput.
- A running synchronous engine cannot be forcibly preempted; stop waits for
  the bounded in-flight call to return. Future engines must provide
  cooperative cancellation before a deadline can be claimed.

### Neutral

- Common 44.1 kHz device configurations are rejected during preflight until a
  separately validated resampler is available.
- M2.2 proves plumbing, time ranges, and failure semantics. It does not claim
  real VAD, whisper.cpp, Metal acceleration, transcript quality, or hardware
  acceptance without the documented manual run.

## Alternatives Considered

**A second ingress consumer for ASR:** Rejected because `try_consume` removes
and recycles a packet, so meter and ASR would receive different packet sets.

**Infer or persist directly from the CPAL callback:** Rejected because model
execution, allocations, database locks, and UI work can block realtime audio.

**Use an unbounded channel and retry later:** Rejected because a slow or absent
model would turn microphone input into unbounded local memory.

**Represent inference overload as `CaptureGap`:** Rejected because it falsely
claims audio was not captured and destroys the distinction needed for review.

**Emit fixture text when no runtime engine is configured:** Rejected because a
real microphone path must not assert a transcription it did not derive.

## References

- [M2.2 native inference bridge plan](../plans/2026-08-10-m2-2-native-inference-bridge.md)
- [Offline speech roadmap](../plans/2026-08-07-word-covenant-roadmap.md)
- [M2.1 local speech pipeline plan](../plans/2026-08-08-m2-1-local-speech-pipeline.md)
