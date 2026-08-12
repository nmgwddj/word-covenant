# ferrous-opencc

- Rust crate: `ferrous-opencc` `0.4.0`
- Source: <https://github.com/apoint123/ferrous-opencc>
- License: Apache-2.0
- Crates.io checksum: `07be899468d0b66213a7fc459fc6ac32db79e397892792e73c8526e8863c7078`
- Enabled feature: `t2s-conversion`

## Runtime boundary

The application uses the pure-Rust, embedded `t2s` dictionaries only when
projecting durable Whisper transcript revisions to Simplified Chinese for the
local UI. The raw Whisper text remains unchanged in SQLite and the audit chain.
Conversion is deterministic, process-local, and performs no network or file
access at runtime.
