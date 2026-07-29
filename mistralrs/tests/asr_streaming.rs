//! The streaming-vs-whole-clip ASR equivalence gate is the `asr_streaming`
//! example (`mistralrs/examples/asr_streaming/main.rs`): it needs a local
//! Voxtral Realtime snapshot and a synthesized WAV, transcribes the clip
//! through both paths, and exits non-zero on transcript mismatch or missing
//! incrementality. This placeholder keeps the gate discoverable from
//! `cargo test`.

#[test]
#[ignore = "run manually: cargo run -p mistralrs --example asr_streaming --features metal --release (needs ASR_STREAMING_MODEL + ASR_STREAMING_WAV)"]
fn asr_streaming_equivalence_via_example() {
    // Evidence is produced by the example binary; see the module docs.
}
