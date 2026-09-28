use anyhow::{ensure, Result};
use mistralrs::{AudioInput, MultimodalModelBuilder, TimedAsrText};

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = std::env::var("ASR_STREAMING_MODEL")?;
    let wav = std::env::var("ASR_STREAMING_WAV")?;
    let audio = AudioInput::from_bytes(&std::fs::read(wav)?)?;
    ensure!(audio.sample_rate == 16000, "fixture must be 16 kHz");
    let samples = audio.to_mono();
    let model = MultimodalModelBuilder::new(model_path).build().await?;
    let mut ordinary = model.asr_stream().await?;
    let mut baseline = String::new();
    for pcm in samples.chunks(1600) {
        baseline.push_str(&ordinary.feed(pcm).await?);
    }
    baseline.push_str(&ordinary.finish().await?);
    let mut previous: Option<Vec<TimedAsrText>> = None;
    for size in [1600, 5120] {
        let mut stream = model.asr_stream().await?;
        let mut timed = Vec::new();
        for pcm in samples.chunks(size) {
            timed.extend(stream.feed_timed(pcm).await?);
        }
        timed.extend(stream.finish_timed().await?);
        let text: String = timed.iter().map(|p| p.text.as_str()).collect();
        ensure!(
            text == baseline,
            "timed text differs from ordinary continuous transcription"
        );
        for p in &timed {
            if let Some(s) = &p.samples {
                ensure!(s.start < s.end && s.end <= samples.len(), "invalid span");
            }
        }
        if let Some(prior) = &previous {
            ensure!(&timed == prior, "timing depends on input batching");
        }
        println!(
            "{}",
            serde_json::json!({"chunk_samples":size,"text":text,"pieces":timed.iter().map(|p| serde_json::json!({"text":p.text,"samples":p.samples})).collect::<Vec<_>>()})
        );
        previous = Some(timed);
    }
    Ok(())
}
