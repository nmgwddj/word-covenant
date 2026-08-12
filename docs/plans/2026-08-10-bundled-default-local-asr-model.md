# Bundled Default Local ASR Model Implementation Plan

> 2026-08-11 update: runtime SHA-256 scanning of the bundled model was removed.
> Release staging and DMG verification still validate the reviewed digest; the
> signed App Bundle is the production runtime trust boundary. SHA-256 checks for
> user-imported external models remain unchanged. The detailed runtime-hash
> steps below are retained only as historical context for the original plan.

**Goal:** Ship one reviewed multilingual Whisper GGML base model inside each
macOS release, make it the verified offline default for a new app run, and
retain a separately imported local model as an advanced temporary override.

**Architecture:** A release-only staging command mirrors an immutable model
lock into Tauri resources alongside `ggml-base.bin`. Native code compares the
packaged manifest with the lock compiled into the app, hashes the regular
resource directly, and creates a native-only capability for the existing
Whisper adapter. The default model is never copied into application data or
recorded as a user import; user-import storage remains the boundary for a
person's separately supplied model. The WebView receives only compact model
metadata and availability state.

**Tech Stack:** Rust, Tauri 2 macOS bundle resources, SHA-256, whisper-rs /
whisper.cpp with Metal, Vue 3, Pinia, Vitest, Cargo tests, GitHub Actions, and
a release-only Node staging command. No desktop HTTP client, model downloader,
telemetry, cloud ASR, raw PCM IPC, speaker embedding, or automatic diarization.

---

## Product Contract

### Reviewed Default Artifact

| Field            | Required value / rule                                             |
| ---------------- | ----------------------------------------------------------------- |
| Artifact         | `ggml-base.bin`, multilingual base, not English-only              |
| Input format     | exactly `whisper.cpp-ggml`                                        |
| Logical identity | stable UUID for byte-identical weights only                       |
| Integrity        | lower-case SHA-256 and exact non-zero byte count                  |
| Provenance       | immutable upstream repository/ref and release-reviewed source URL |
| Licence evidence | model-card ID and MIT licence from the reviewed upstream card     |
| Release location | `WordCovenant.app/Contents/Resources/models/`                     |

The current reviewed model is `ggml-base.bin` at 147,951,465 bytes with SHA-256
`60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe`.
The binary is ignored by Git; it is never committed to the main repository or
stored through Git LFS.

Developer ID signing and notarization remain M6 release gates. This model
integration validates local bytes and package consistency; it does not claim
that an unsigned development or draft-release bundle is publisher-attested.

### Egress Boundary

The installed application has no model download, update, recovery, telemetry,
or cloud fallback path. It stays offline whether or not the visible egress
toggle is later enabled for an unrelated approved tool. Only an explicit
release build command or CI step may fetch the fixed upstream artifact. Normal
local builds consume a supplied verified file and fail clearly if it is absent.

### Startup And Recording

At each launch, before microphone capture is possible:

1. Tauri provides the native resource directory; it is never passed across IPC.
2. Rust parses the packaged `models/manifest.json` and requires an exact match
   with the model lock compiled into the executable.
3. Rust rejects an unsafe resource root, symlink, missing file, non-regular
   file, wrong model kind/format, size mismatch, or SHA-256 mismatch.
4. A matching resource produces compact native-only verified metadata; its
   path and bytes remain in Rust. Engine loading obtains a separate verified
   byte buffer immediately before Whisper initialization.
5. The built-in model becomes the app-run default. A person still explicitly
   presses Record before macOS microphone access begins.
6. Whisper rehashes the same resource immediately before every engine load.

If verification fails, the desktop still opens and displays a safe local error.
Recording is disabled until a person visibly selects a separately verified
advanced local model. The failure never triggers a network request, a model
retry, synthetic text, or a system/cloud recognizer fallback.

### User Experience

The flat white/gray model panel shows `内置默认` as ready and selected when the
resource validates. A plus icon opens the existing advanced local-import flow.
Imported compatible models can override the selection only for the current
process; a fresh launch returns to the verified built-in default.

This change is solely an ASR first-use improvement. It does not provide speaker
diarization, voiceprint matching, named-person recognition, overlap handling,
or an accuracy guarantee.

## Task 1: Lock And Stage The Release Artifact

**Files:**

- Create: `models/whisper-base.lock.json`
- Create: `src-tauri/resources/models/manifest.json`
- Create: `scripts/stage-bundled-model.mjs`
- Create: `scripts/stage-bundled-model.test.mjs`
- Modify: `src-tauri/tauri.conf.json`, `.gitignore`, `package.json`

1. Store reviewable metadata in the lock: schema version, UUID, GGML format,
   multilingual/base flags, artifact name, byte count, SHA-256, model card,
   licence, timestamp, and source repository/ref/URL.
2. Mirror the lock into the packaged manifest. The staging command rejects any
   differing field, so a hand-edited manifest cannot be bundled accidentally.
3. Require an explicit local source file. Stream it into the ignored resource
   path while calculating SHA-256 and byte count; use a temporary file, a
   `0644` final resource mode, and atomic rename after validation.
4. Keep the staging command offline: it accepts no URL or fetch flag. The
   release workflow alone may explicitly fetch the immutable URL in the
   reviewed lock, then invokes the same local staging command. Never use
   user-provided URLs, credentials, or runtime application code.
5. Add exact `bundle.resources` mappings for the model, its manifest, and its
   license notice. Do not use a broad repository glob.
6. Test good staging, manifest divergence, missing source, hash mismatch,
   wrong size, non-regular source, final readable permissions, and a failed
   temporary write.

**Verification:**

```sh
node --test scripts/stage-bundled-model.test.mjs
pnpm model:verify
```

## Task 2: Validate The Packaged Resource In Rust

**Files:**

- Create: `src-tauri/src/inference/bundled_model.rs`
- Modify: `src-tauri/src/inference/mod.rs`
- Modify: `src-tauri/src/inference/model_registry.rs`

1. Compile the reviewed lock into the native binary with `include_str!`.
2. Parse a packaged manifest from the native resource root and require typed
   equality with the compiled lock before processing the artifact.
3. Validate only normal relative path components and reject a symlinked root,
   manifest, directory component, or artifact. Require a regular non-empty
   `models/ggml-base.bin` below the canonical resource root.
4. Stream SHA-256 and size verification without reading the weights into an
   IPC object or a whole-file memory buffer.
5. Add a crate-private `VerifiedModelArtifact` constructor for this already
   verified bundle boundary. It must not be callable from a Tauri command or
   serialize an absolute filesystem path.
6. Test good fixtures, wrong hash/size, missing data, malformed metadata,
   incompatible input format, non-multilingual flag, unsafe paths, symlinks,
   and the embedded lock itself. Tests must make no network call.

**Verification:**

```sh
cargo test --manifest-path src-tauri/Cargo.toml bundled_model --lib
cargo fmt --manifest-path src-tauri/Cargo.toml --check
```

## Task 3: Select The Bundle Without Mutating User-Import Evidence

**Files:**

- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src-tauri/src/commands.rs`

1. Keep a native-only bundled-runtime object containing the resource directory,
   current verified metadata, and a small serializable availability status.
2. During Tauri setup, resolve the resource directory and initialize that object
   before the command handler becomes available. A failure keeps the app open
   with an unavailable status rather than using startup network recovery.
3. Add the bundled model as a virtual first model in the list projection only
   while it is verified. Do not insert it into `local_models` or create a
   `LocalModelImported` audit event; those records retain their truthful
   user-acknowledgement meaning.
4. Set the per-run `ActiveLocalAsrProfile` to the built-in UUID after success.
   Existing imported models may be selected visibly while not recording. On a
   restart the built-in default is selected again.
5. Resolve built-in IDs through the bundled runtime and all other IDs through
   the existing user-import registry. Before microphone preparation, rehash the
   selected bundle and map errors to a fixed Chinese message without a path.
6. Keep provider, model version, and SHA-256 in final-transcript provenance.

**Verification:**

```sh
cargo test --manifest-path src-tauri/Cargo.toml state commands --lib
cargo check --manifest-path src-tauri/Cargo.toml --release
```

## Task 4: Project Default And Advanced Override In The UI

**Files:**

- Modify: `src/types.ts`, `src/lib/wordCovenantApi.ts`, `src/stores/models.ts`
- Modify: `src/App.vue`, `src/components/ModelRegistryPanel.vue`
- Modify: `src/assets/main.css`
- Test: corresponding Vitest files

1. Add a `get_bundled_asr_status` command projection with `available`, opaque
   `modelId`, and a fixed safe message. Browser preview returns an unavailable
   local-only state and never fetches anything.
2. Initialize model registrations, active profile, and bundled status together.
3. Present the active bundle as `内置默认 · 已启用`; present imported compatible
   models as `高级本地模型`. Preserve the selector and block changes during
   recording.
4. When the default is unavailable, show a local error and leave imported
   models as the only selectable fallback. Do not offer download links, source
   URLs, a model gallery, or network toggles.
5. Maintain the restrained flat layout: white/gray surfaces, compact metadata,
   familiar icons, no nested cards, and no new decorative borders.
6. Test ready default, valid advanced override, unavailable default with and
   without fallback, browser preview, no path rendering, and narrow viewport
   text fitting.

**Verification:**

```sh
pnpm vitest run src/stores/models.spec.ts src/components/ModelRegistryPanel.spec.ts src/lib/wordCovenantApi.spec.ts
pnpm type-check
pnpm build
```

## Task 5: Release And Offline Acceptance

**Files:**

- Modify: `.github/workflows/release.yml`, `.github/workflows/test-build.yml`,
  `.github/workflows/test.yml`
- Create: `src-tauri/resources/third-party/whisper-base-model-MIT.txt`
- Modify: `README.md`, `docs/third-party/whisper-rs.md`,
  `docs/plans/2026-08-10-m2-3-real-local-speech-acceptance.md`

1. Restrict model-bearing bundles to macOS runners.
2. Make release and package CI explicitly fetch the lock-pinned artifact,
   verify its byte count and SHA-256, then invoke the local staging command
   before Tauri build; unit-test CI verifies lock/manifest/staging behavior
   without downloading the artifact.
3. Independently verify that exactly one artifact and one matching manifest are
   present under `Contents/Resources/models/`.
4. Record source/ref, model-card identifier, MIT licence, byte count, SHA-256,
   app version, signing, and notarization evidence. Do not claim a redistribution
   right without the release owner's licence review.
5. Run on a clean macOS installation with egress disabled and process-level
   network monitoring. Confirm default readiness, user-triggered microphone
   permission, Chinese final transcripts with capture times, tampered/missing
   resource failure, advanced override, and zero outbound connections.

**Verification:**

```sh
pnpm model:verify
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
pnpm tauri build --bundles app
git diff --check
```

## Release Acceptance Criteria

- Lock, packaged manifest, and staged artifact SHA-256 match during packaging;
  runtime checks only the signed bundle manifest, regular-file type, and size.
- A fresh offline macOS install exposes the verified default immediately and
  reaches microphone-to-final-transcript only after the user presses Record.
- Missing, symlinked, altered, or malformed resources never reach Whisper or
  microphone preparation, and they never initiate download or cloud fallback.
- Imported compatible models keep their existing local SHA/licence contract and
  can override only for the current run.
- No path, model bytes, PCM, or automatic-person-identification claim appears
  in the WebView, SQLite/audit payloads, logs, or network traffic.
