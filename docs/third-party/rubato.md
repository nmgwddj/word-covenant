# rubato

- Rust crate: `rubato` `0.16.2`
- Source: <https://github.com/HEnquist/rubato>
- License: MIT
- Crates.io checksum: `5258099699851cfd0082aeb645feb9c084d9a5e1f1b8d5372086b989fc5e56a1`

## Runtime boundary

The application uses `rubato` locally to convert 48 kHz microphone PCM to the
16 kHz mono format required by VAD and Whisper. Resampling runs on the bounded
native dispatcher, outside the CoreAudio callback. PCM remains process-local
and is not serialized, persisted, logged, or sent over the network.
