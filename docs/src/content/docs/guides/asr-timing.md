---
title: Continuous ASR audio timing
---

Voxtral `AsrStream::feed_timed` and `finish_timed` return committed text pieces
with optional nominal 16 kHz sample ranges. Select timed mode on the first
feed. The fixed decoder audio positions advance by 80 ms per token, after
accounting for the model prompt and delay. They are approximate token times,
not forced-aligned word boundaries. Control tokens emit no text; incomplete
UTF-8 suffixes wait for completion. Right-padding-only output has no range.
Ordinary `feed` / `finish` callers retain their existing text path.

Run `cargo run -p mistralrs --example asr_timing --features metal` with
`ASR_STREAMING_MODEL` pointing to a local Voxtral snapshot and
`ASR_STREAMING_WAV` pointing to a 16 kHz WAV. It requires timed text to equal
the ordinary continuous transcript and requires identical pieces and ranges
under 100 ms and 320 ms input batching. This is an API equivalence check, not
an acoustic word-alignment accuracy claim.
