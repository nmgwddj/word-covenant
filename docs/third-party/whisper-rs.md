# whisper-rs / whisper.cpp

- Rust crate: `whisper-rs` `0.16.0`
- Rust FFI crate: `whisper-rs-sys` `0.15.0`
- Source: <https://codeberg.org/tazz4843/whisper-rs>
- Upstream runtime: <https://github.com/ggml-org/whisper.cpp>
- `whisper-rs` and `whisper-rs-sys` license: Unlicense
- Bundled whisper.cpp / ggml license: MIT
- macOS build feature: `metal`

## Local model compatibility

This release accepts exactly one imported ASR artifact declaration:

```text
whisper.cpp-ggml
```

`whisper-rs` 0.16 bundles a whisper.cpp model loader that validates the GGML
file magic. It does not load GGUF artifacts in this integration. The filename
extension is not used as format evidence.

The application never downloads a model. A person obtains the compatible
multilingual Whisper GGML model outside the application, verifies its SHA-256,
and explicitly confirms its model card and license before local import. The
model artifact's own license and model-card terms remain the importer's
responsibility and are recorded with the local registration.

## Runtime boundary

The adapter receives 16 kHz mono PCM only in native Rust memory. It disables
translation and automatic language selection, emits final transcript records
with registered model provenance, and has no HTTP client, model URL, cloud
fallback, or WebView PCM transport.
