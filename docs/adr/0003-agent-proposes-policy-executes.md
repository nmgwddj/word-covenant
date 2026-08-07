# ADR-0003: Separate Agent proposal from policy-governed execution

## Status

Accepted

## Context

Agent planners, skills, hooks, transcripts, and remote responses are untrusted for authority purposes. Allowing them to directly select a URL, start a process, or run a tool would make prompt injection and plugin compromise side-effecting security failures.

## Decision

All planners and extensions return typed `PlanV1` / `ActionProposal` values. The Rust policy engine and tool broker own validation, consent, execution, and audit. Initial tools are closed-set local TTS, notification, and fixed HTTP profiles; arbitrary shell execution and generic hooks are excluded.

## Consequences

### Positive

- The execution surface is testable, auditable, and narrow.
- Prompt injection cannot directly create authority.

### Negative

- New capabilities require a manifest/tool implementation instead of free-form agent code.
- Plugin development is deliberately slower until a constrained extension mechanism exists.

## Alternatives Considered

**Agent framework executes tools directly:** Rejected because it conflates text generation with authority.

**Native sidecar plugins in MVP:** Rejected because sidecars run with user privileges and are not a sufficient sandbox.
