# ADR-0002: Start speaker workflow with anonymous clusters

## Status

Accepted

## Context

Speaker segmentation/clustering answers whether two segments may be the same voice. It does not reliably establish a person's identity. Mapping an ambient voice to a name is sensitive biometric processing with heightened error, consent, and legal risk.

## Decision

The product initially displays anonymous `Speaker N` clusters, uncertainty, and overlap. Users may correct, merge, split, and optionally rename clusters. Voiceprint enrolment, identity matching, and profile persistence require a later explicit design, consent flow, deletion path, and quality threshold.

## Consequences

### Positive

- MVP delivers useful conversation structure without making unjustified identity claims.
- The stored data is easier to minimise and delete.

### Negative

- Users must manually identify speakers where that matters.
- Product differentiation based on named speaker recognition arrives later.

## Alternatives Considered

**Automatic speaker naming from voiceprints:** Rejected for MVP due to biometric sensitivity, false positives, and consent obligations.

**No diarization at all:** Rejected because anonymous clusters materially improve timeline usability and can be designed with uncertainty.
