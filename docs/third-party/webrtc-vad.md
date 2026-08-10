# webrtc-vad / libfvad

- Rust crate: `webrtc-vad` `0.4.0`
- Source: <https://github.com/kaegi/webrtc-vad>
- Rust wrapper license: MIT
- Bundled libfvad source: <https://github.com/dpirch/libfvad>
- Bundled libfvad license: BSD 3-Clause

## Runtime boundary

The application uses libfvad through the Rust wrapper for exact 10 ms, 16 kHz
mono frames. The selected runtime setting is `Aggressive`; it is a speech gate,
not a speaker-identity, voiceprint, or diarization system.

libfvad is built into the application dependency graph and requires no model
file, account, HTTP request, or download at runtime. Temporary `i16` PCM is
created only inside the native detector call, then dropped. It is not
serialized, persisted, logged, sent through Tauri IPC, or exposed to the
WebView.
