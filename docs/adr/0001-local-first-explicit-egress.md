# ADR-0001: Require an explicit visible egress gate

## Status

Accepted

## Context

WordCovenant handles room audio, transcripts, speaker-derived data, and Agent context. An approval object alone can be created through a command or restored from data, so it does not by itself prove that the person at the UI currently intends to allow data to leave the device.

## Decision

Network egress uses three mandatory checks: a session-only Rust-held master switch defaulting to disabled after startup, a user-approved exact tool/profile and HTTPS origin with data categories/duration, and a final policy evaluation immediately before connection creation. The visible UI may request the switch change but never substitutes for the Rust check. Turning the switch off or revoking approval blocks later requests immediately.

## Consequences

### Positive

- A clear product state satisfies the local-first promise and is easy to audit.
- Approval records cannot silently grant networking after restart.
- Agent, plugin, and WebView authority stays subordinate to policy.

### Negative

- Users must make two intentional choices for first use of an external tool.
- Unattended background sync is excluded unless a later design explicitly changes the product contract.

### Neutral

- Tauri CSP/capabilities remain restrictive even after the switch is enabled; a future Rust executor owns the outbound client.

## Alternatives Considered

**Persisted per-profile approval only:** Rejected because it can lead to surprising background egress and lacks an obvious current user-intent signal.

**Front-end-only toggle:** Rejected because a compromised WebView, IPC misuse, or future plugin could bypass it.

**Global permanent online mode:** Rejected because it weakens the local-first boundary and makes audit/recovery ambiguous.
