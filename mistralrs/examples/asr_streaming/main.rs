//! Equivalence gate for the Voxtral streaming ASR session.
//!
//! Transcribes one WAV twice — once through the whole-clip request path and
//! once through `Model::asr_stream()` fed in 100 ms chunks — and exits
//! non-zero unless the two transcripts match (after trimming whitespace) AND
//! at least one non-empty text delta arrived before the final chunk was fed.
//!
//! Usage:
//! ```bash
//! say -v Samantha -o /tmp/asr-fixture.aiff "The quick brown fox jumps over \
//!   the lazy dog while the committee reviews the annual budget."
//! afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/asr-fixture.aiff /tmp/asr-fixture.wav
//! ASR_STREAMING_WAV=/tmp/asr-fixture.wav \
//! ASR_STREAMING_MODEL=/path/to/Voxtral-Mini-4B-Realtime-2602/snapshot \
//!   cargo run -p mistralrs --example asr_streaming --features metal --release
//! ```

use anyhow::{bail, Context, Result};
use mistralrs::{AudioInput, MultimodalMessages, MultimodalModelBuilder, TextMessageRole};

/// 100 ms at 16 kHz.
const CHUNK_SAMPLES: usize = 1600;

#[tokio::main]
async fn main() -> Result<()> {
    let model_id = std::env::var("ASR_STREAMING_MODEL")
        .unwrap_or_else(|_| "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string());
    let wav_path =
        std::env::var("ASR_STREAMING_WAV").unwrap_or_else(|_| "/tmp/asr-fixture.wav".to_string());

    let audio_bytes = std::fs::read(&wav_path).with_context(|| format!("reading {wav_path}"))?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;
    if audio.sample_rate != 16000 {
        bail!(
            "fixture must be 16 kHz (got {} Hz); the streaming session takes raw 16 kHz samples",
            audio.sample_rate
        );
    }
    let samples = audio.to_mono();
    println!(
        "Loaded {wav_path}: {} samples ({:.2} s)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );

    let model = MultimodalModelBuilder::new(&model_id)
        .with_logging()
        .build()
        .await?;

    // ---- Whole-clip reference (deterministic sampling is the default) ----
    let messages = MultimodalMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Transcribe this audio.",
        vec![],
        vec![audio],
        vec![],
    );
    let response = model.send_chat_request(messages).await?;
    let whole_clip = response.choices[0]
        .message
        .content
        .clone()
        .context("whole-clip transcription returned no content")?;
    println!("\n=== whole-clip transcript ===\n{whole_clip}\n");

    // ---- Streaming: 100 ms chunks ----
    let mut stream = model.asr_stream().await?;
    let mut streamed = String::new();
    let mut deltas: Vec<(usize, String)> = Vec::new();
    let n_chunks = samples.len().div_ceil(CHUNK_SAMPLES);
    let mut delta_before_last_chunk = false;
    let mut window_started = std::time::Instant::now();
    let mut window_means: Vec<f64> = Vec::new();
    for (i, chunk) in samples.chunks(CHUNK_SAMPLES).enumerate() {
        let delta = stream.feed(chunk).await?;
        // Real-time budget check: each chunk is 100 ms of audio, so the
        // mean feed cost per 100-chunk window must stay well under 100 ms
        // or a live session falls ever further behind the speaker.
        if (i + 1) % 100 == 0 {
            let elapsed = window_started.elapsed();
            let mean_ms = elapsed.as_secs_f64() * 10.0;
            println!(
                "pace  {:>4}: mean feed cost {:>6.1} ms over last 100 chunks",
                i + 1,
                mean_ms
            );
            window_means.push(mean_ms);
            window_started = std::time::Instant::now();
        }
        if !delta.is_empty() {
            println!("feed {:>3}/{n_chunks}: {delta:?}", i + 1);
            if i + 1 < n_chunks {
                delta_before_last_chunk = true;
            }
            streamed.push_str(&delta);
            deltas.push((i + 1, delta));
        }
    }
    let tail = stream.finish().await?;
    if !tail.is_empty() {
        println!("finish       : {tail:?}");
        streamed.push_str(&tail);
    }
    println!("\n=== streaming transcript ===\n{streamed}\n");

    // ---- The gate ----
    let mut failed = false;
    if whole_clip.trim() != streamed.trim() {
        eprintln!("MISMATCH between whole-clip and streaming transcripts");
        failed = true;
    } else {
        println!("MATCH: streaming transcript equals whole-clip transcript");
    }
    if !delta_before_last_chunk {
        eprintln!("NOT INCREMENTAL: no non-empty delta arrived before the final chunk");
        failed = true;
    } else {
        println!(
            "INCREMENTAL: first delta arrived at feed {}/{n_chunks}",
            deltas.first().map(|(i, _)| *i).unwrap_or(0)
        );
    }
    // Flat-pace gate for long clips: per-feed cost must not grow with session
    // length (a dictation session of 30+ minutes has to keep real-time pace).
    // With more than 1000 chunks, the last full 100-chunk window may cost at
    // most 1.5x the warm baseline (windows 2-4; window 1 is warm-up).
    if n_chunks > 1000 && window_means.len() >= 4 {
        let baseline = window_means[1..4].iter().sum::<f64>() / 3.0;
        let last = *window_means
            .last()
            .expect("window_means has at least 4 entries");
        if last > 1.5 * baseline {
            eprintln!(
                "PACE NOT FLAT: last window mean {last:.1} ms exceeds 1.5x warm baseline {baseline:.1} ms"
            );
            failed = true;
        } else {
            println!(
                "PACE FLAT: last window mean {last:.1} ms within 1.5x warm baseline {baseline:.1} ms"
            );
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}
