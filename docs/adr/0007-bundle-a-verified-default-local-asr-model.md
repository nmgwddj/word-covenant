# ADR-0007: Bundle a release-verified default local ASR model

## Status

Accepted

## Context

M2.3 has a functioning local Whisper pipeline, but its first-use path requires
the person using WordCovenant to obtain, hash, import, acknowledge, and select
a compatible model before a microphone session can start. That is a useful
advanced-model boundary, but it delays the first real local recording on a
clean macOS installation.

The client must remain local-first. Model availability, release verification, loading,
VAD, and ASR must not create a network request. The session egress switch
remains disabled by default and is not implicit permission for model download,
update, telemetry, or cloud fallback. A release artifact must also be resistant
to an accidental, missing, or substituted model file, rather than treating a
filename or extension as evidence.

The default needs to transcribe Chinese without a first-use download. The
chosen multilingual Whisper GGML `large-v3-turbo-q5_0` weights improve the
Chinese and mixed-language baseline while remaining practical on supported
Apple Silicon Macs. The current adapter explicitly decodes `zh`; it does not
offer automatic language detection or guarantee every mixed-language phrase.

## Decision

Each distributable macOS build bundles exactly one manifest-defined multilingual
`ggml-large-v3-turbo-q5_0.bin` artifact for the existing `whisper.cpp-ggml` adapter. It is the
standard local transcription model, not a speaker-diarization or voiceprint
model. The manifest is packaged alongside the artifact and contains a schema
version, immutable model UUID, model kind, `whisper.cpp-ggml` input format,
variant/version, exact filename, byte count, SHA-256, model-card identifier,
license identifier, and release source/provenance. A distributable manifest
must never contain a placeholder digest.

The release staging job obtains or receives the artifact before the Tauri
bundle is built, calculates its SHA-256, compares it to the reviewed manifest,
and fails closed on every mismatch. Any staging action that fetches an upstream
artifact is an explicit release-only operation, outside the desktop client; it
is disabled by default and cannot be reached from application code. The signed
and notarized application package protects the manifest and model after release
engineering configures Developer ID signing. Runtime code deliberately does not
rehash the large bundled artifact. Unsigned development builds rely on the same
reviewed staging output but do not provide a signature-backed trust guarantee.

At startup, native code resolves the packaged resource directory, validates the
manifest and bundled regular file without following a symlink, checks the byte
count, and projects compact model metadata without reading the model bytes. It compares
the packaged manifest with an immutable lock compiled into the native binary.
Immediately before Whisper loads, native code reads the artifact through one
no-follow file descriptor and passes that buffer to the adapter without
computing a runtime digest. It does not copy the model into application data
or register publisher-supplied bytes as a user import. A bundled model is never
accepted by name, path, extension, or cached metadata alone.

When that metadata check succeeds, the active ASR profile for this application
run defaults visibly to the bundled model. Recording still requires the person
to press the existing record control; selecting a local default does not start
capture or enable egress. If the resource is unavailable or loading fails, the UI shows
the default as unavailable and microphone capture is blocked unless the person
explicitly selects a separately verified compatible imported model. There is no
runtime redownload, silent repair, cloud fallback, or automatic replacement.

Imported compatible models remain an optional advanced local override. They
keep the current local-file picker, user-provided trusted SHA-256, model-card
and license acknowledgement, managed copy, verification, and explicit
selection contract. They cannot overwrite the bundled artifact or change the
default for a later app launch. At the next launch a valid bundled model is
again the initial active profile; an imported model may replace it only through
a visible choice before recording.

The bundled artifact's license and model-card provenance are release evidence,
not a fabricated user acknowledgement. The user-import registry and its audit
events remain reserved for advanced local imports. Every resulting transcript
still records the bundled model's provider, version, and SHA-256 provenance.
Native paths, model bytes, PCM, and model contents remain outside Tauri IPC,
the WebView, audit payloads, and logs.

## Consequences

### Positive

- A clean supported macOS installation can record and transcribe locally after
  microphone permission, without a model download or manual import.
- The selected default is inspectable by version, format, provenance, size, and
  SHA prefix, while the underlying file path stays native-only.
- Release-time SHA-256 checks and the signed application bundle bind the reviewed
  model to the distributed product without a costly runtime scan.
- An advanced model remains available for users who need a different local
  quality, speed, or language trade-off.

### Negative

- The macOS application grows by roughly the model artifact size, and loading
  Whisper still incurs a one-time memory and initialization cost.
- Model upgrades require a new reviewed, signed application release and a
  compatibility/rollback plan; the client cannot silently self-update weights.
- The bundled model establishes an out-of-box ASR path only. It does not establish
  automatic speaker separation, named-person recognition, overlap handling, or
  a quality guarantee on every device.

### Neutral

- Existing user-imported model files remain local and usable; a valid imported
  advanced model is a deliberate per-run override, not a migration target.
- Release engineering may use a controlled networked fetch before packaging,
  but the shipped application continues to have no model network endpoint or
  HTTP client path for this feature.
- This change does not configure Developer ID signing or notarization. The
  release workflow remains a draft-release path until M6 release operations
  supply those credentials and clean-machine Gatekeeper verification.

## Alternatives Considered

**Require every user to import a model:** Rejected for the default experience
because it prevents immediate verification of real microphone recording on a
new installation. It remains the advanced-model path.

**Download a model on first launch:** Rejected because it violates the
local-first runtime boundary, produces an unreliable first-use dependency, and
would need a separate explicit egress and supply-chain design.

**Copy the bundle into the user-import model registry:** Rejected because it
duplicates the model on disk and would incorrectly make publisher-provided
license evidence look like a user import. The native-only bundled capability
keeps the same byte-level verification without changing user-import history.

**Use a system or cloud recognizer as a fallback:** Rejected because language
resources, model provenance, execution locality, and egress behavior would no
longer be controlled by this application.

**Make an imported advanced model persist as the next-launch default:** Rejected
because it makes the first-visible model implicit and obscures which artifact a
new recording will use. Per-run visible selection is sufficient for an override.

## References

- [ADR-0001: Require an explicit visible egress gate](0001-local-first-explicit-egress.md)
- [ADR-0005: Use a single native dispatcher and bounded inference outcomes](0005-native-dispatcher-bounded-inference-bridge.md)
- [M2.3 real local speech acceptance checklist](../plans/2026-08-10-m2-3-real-local-speech-acceptance.md)
