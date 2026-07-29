//! Incremental (streaming) ASR session for the Voxtral Realtime model.
//!
//! The whole-clip path processes a finished clip in one shot: pad the audio with
//! silence, compute the mel spectrogram, encode it, and autoregressively decode
//! one text token per audio-embedding position. Every stage of that pipeline is
//! causal, so the same result can be produced incrementally:
//!
//! - mel frames are final once their sample window is available
//!   ([`MelFrameEngine::ready_frames`]),
//! - conv frames only look left (causal convolutions, recomputed over the full
//!   mel history and sliced),
//! - encoder frames only attend left (appending KV cache),
//! - adapter groups are independent per group of 4 encoder frames,
//! - a decoder forward at position `p` only needs audio embeddings `..=p`.
//!
//! The session drives exactly the same decoder entry point as whole-clip
//! generation (the audio-embedding-conditioned generation branch of
//! `VoxtralModel::inner_forward`), with the same prompt layout, the same
//! greedy sampling, and the same generation cap, so for the same total sample
//! stream `feed(..)*; finish()` reproduces the whole-clip transcript.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Tensor};
use tokenizers::Tokenizer;

use crate::pipeline::{
    text_models_inputs_processor::FlashParams, ModelForwardContext, MultimodalModel, Pipeline,
};

use super::audio_processing::{
    MelFrameEngine, VoxtralAudioProcessor, N_LEFT_PAD_TOKENS, N_RIGHT_PAD_TOKENS,
};
use super::inputs_processor::{
    AUDIO_LENGTH_PER_TOK, BOS_TOKEN_ID, N_DELAY_TOKENS, STREAMING_PAD_TOKEN_ID,
};
use super::{VoxtralModel, VoxtralSpecificArgs};

/// Decoder prompt length: `[BOS]` + `(32 left-pad + 6 delay)` streaming pads.
const N_PROMPT_TOKENS: usize = 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS;

/// A streaming ASR session on a loaded Voxtral Realtime model.
///
/// Exactly one session may exist per loaded model at a time: the session owns
/// the model's engine-global state (encoder KV cache, decoder KV cache, audio
/// embedding cache) for its lifetime, and no other request may run against the
/// model while the session is active. Creating a session resets that state, so
/// an abandoned prior session cannot poison a new one.
pub struct VoxtralAsrSession {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    tokenizer: Arc<Tokenizer>,
    eos_toks: Vec<u32>,
    mel_engine: MelFrameEngine,
    num_mel_bins: usize,
    samples_per_token: usize,
    /// Raw 16 kHz mono sample buffer; starts with the 32-token left pad of silence.
    samples: Vec<f32>,
    /// All mel frames computed so far, flattened `[frame][mel_bin]`.
    mel_flat: Vec<f32>,
    /// Number of mel frames in `mel_flat`.
    mel_done: usize,
    /// Conv frames already pushed through the encoder transformer (== the
    /// total ever appended to its KV cache; the cache retains only the last
    /// 750 once the sliding window saturates).
    conv_done: usize,
    /// Encoder output frames awaiting a complete adapter group of 4.
    enc_pending: Option<Tensor>,
    /// Accumulated audio embeddings `[1, N, dim]`.
    embeds: Option<Tensor>,
    n_embeds: usize,
    downsample: usize,
    prompt_fed: bool,
    /// Number of decoder positions consumed (prompt + generated tokens fed back).
    tokens_fed: usize,
    /// Generated token ids (EOS excluded), in order.
    gen_ids: Vec<u32>,
    hit_eos: bool,
    /// Decoded text already returned to the caller.
    emitted: String,
}

fn voxtral_of(pipeline: &dyn Pipeline) -> Result<&VoxtralModel> {
    pipeline
        .multimodal_model()
        .and_then(|model| model.as_voxtral())
        .ok_or_else(|| anyhow!("streaming ASR requires a loaded Voxtral Realtime model"))
}

/// Run the decoder over `ids` at absolute positions `pos..pos + ids.len()`
/// through the audio-conditioned generation branch, returning the logits of the
/// last position.
fn forward_ids(model: &VoxtralModel, ids: &[u32], pos: usize) -> Result<Tensor> {
    let device = MultimodalModel::device(model);
    let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?;
    let seqlen_offsets = [pos];
    let context_lens = [(ids.len() - 1, 1)];
    let position_ids = [pos + ids.len()];
    // No flash-attn metadata: Voxtral runs the mask-based SDPA path.
    let flash_params = FlashParams::empty(true);
    let mut ctx = ModelForwardContext::new(
        &seqlen_offsets,
        &context_lens,
        &position_ids,
        None,
        &flash_params,
    );
    let logits = MultimodalModel::forward(
        model,
        &input,
        None,
        Box::new(VoxtralSpecificArgs {
            mel_features: None,
            n_delay_tokens: Some(N_DELAY_TOKENS as f32),
        }),
        &mut ctx,
    )?;
    Ok(logits)
}

fn argmax_id(logits: &Tensor) -> Result<u32> {
    Ok(logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .argmax(0)?
        .to_scalar::<u32>()?)
}

impl VoxtralAsrSession {
    /// Open a session. Resets the model's session state (audio embedding cache,
    /// encoder KV cache, decoder KV cache) before use.
    pub async fn new(pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>) -> Result<Self> {
        let (tokenizer, eos_toks, mel_engine, num_mel_bins, samples_per_token, downsample) = {
            let guard = pipeline.lock().await;
            let model = voxtral_of(&*guard)?;

            // A crashed or abandoned prior session must not poison this one.
            model.reset_model_specific_state();
            model.reset_decoder_cache();

            let tokenizer = guard
                .tokenizer()
                .ok_or_else(|| anyhow!("Voxtral pipeline has no tokenizer"))?;
            let eos_toks = guard.get_metadata().eos_tok.clone();

            let audio_cfg = model.audio_encoding_args();
            let processor = VoxtralAudioProcessor::new(audio_cfg);
            let mel_engine = processor.frame_engine()?;
            (
                tokenizer,
                eos_toks,
                mel_engine,
                audio_cfg.num_mel_bins,
                processor.samples_per_token(),
                model.adapter_downsample_factor(),
            )
        };

        Ok(Self {
            pipeline,
            tokenizer,
            eos_toks,
            mel_engine,
            num_mel_bins,
            samples_per_token,
            samples: vec![0.0f32; N_LEFT_PAD_TOKENS * samples_per_token],
            mel_flat: Vec::new(),
            mel_done: 0,
            conv_done: 0,
            enc_pending: None,
            embeds: None,
            n_embeds: 0,
            downsample,
            prompt_fed: false,
            tokens_fed: 0,
            gen_ids: Vec::new(),
            hit_eos: false,
            emitted: String::new(),
        })
    }

    /// Append 16 kHz mono f32 samples as they arrive. Returns whatever text the
    /// model newly committed, in order (possibly empty).
    pub async fn feed(&mut self, pcm: &[f32]) -> Result<String> {
        let pipeline = Arc::clone(&self.pipeline);
        let guard = pipeline.lock().await;
        let model = voxtral_of(&*guard)?;

        self.samples.extend_from_slice(pcm);
        self.advance_mel(false);
        self.advance_encoder(model)?;
        self.advance_decoder(model, None)?;
        self.take_delta()
    }

    /// The speaker stopped: append the right-pad silence, drain generation to
    /// the whole-clip cap, return the final tail text, and reset the model state.
    pub async fn finish(mut self) -> Result<String> {
        let pipeline = Arc::clone(&self.pipeline);
        let guard = pipeline.lock().await;
        let model = voxtral_of(&*guard)?;

        self.samples.extend(std::iter::repeat_n(
            0.0f32,
            N_RIGHT_PAD_TOKENS * self.samples_per_token,
        ));
        self.advance_mel(true);
        self.advance_encoder(model)?;

        // Whole-clip generation cap: ceil(mel_frames / 8) - right-pad tokens.
        let cap = self
            .mel_done
            .div_ceil(AUDIO_LENGTH_PER_TOK)
            .saturating_sub(N_RIGHT_PAD_TOKENS);
        self.advance_decoder(model, Some(cap))?;

        let tail = self.take_delta()?;

        // Leave the model clean for whatever runs next.
        model.reset_model_specific_state();
        model.reset_decoder_cache();

        Ok(tail)
    }

    /// Compute all mel frames that are final. During streaming only frames whose
    /// full sample window exists are computed; at finish the remaining frames are
    /// computed with reflection at the (now final) clip end, exactly like the
    /// whole-clip path.
    fn advance_mel(&mut self, clip_complete: bool) {
        let target = if clip_complete {
            self.mel_engine.total_frames(self.samples.len())
        } else {
            self.mel_engine.ready_frames(self.samples.len())
        };
        if target > self.mel_done {
            let new = self
                .mel_engine
                .compute_frames(&self.samples, self.mel_done, target, clip_complete);
            self.mel_flat.extend(new);
            self.mel_done = target;
        }
    }

    /// Push newly final conv frames through the encoder transformer and complete
    /// adapter groups into the accumulated audio embeddings.
    fn advance_encoder(&mut self, model: &VoxtralModel) -> Result<()> {
        // Conv frame j depends on mel frames ..=2j+1, so with M mel frames the
        // first M/2 conv frames are final. The convolutions are cheap; recompute
        // them over the full history and slice the new frames for exactness.
        let target_conv = self.mel_done / 2;
        if target_conv > self.conv_done {
            let device = MultimodalModel::device(model);
            let mel = Tensor::from_vec(
                self.mel_flat.clone(),
                (1, self.mel_done, self.num_mel_bins),
                device,
            )?;
            let conv_all = model.encoder.convolve(&mel)?;
            let new = conv_all
                .narrow(1, self.conv_done, target_conv - self.conv_done)?
                .contiguous()?;
            let enc_out = model.encoder.encode_new(&new)?;
            self.conv_done = target_conv;
            self.enc_pending = Some(match self.enc_pending.take() {
                Some(prev) => Tensor::cat(&[&prev, &enc_out], 1)?,
                None => enc_out,
            });
        }

        // Adapter: complete groups of `downsample` encoder frames only; the
        // remainder waits (whole-clip drops a trailing partial group the same way).
        if let Some(pending) = self.enc_pending.take() {
            let n = pending.dim(1)?;
            let n_grouped = n - n % self.downsample;
            if n_grouped > 0 {
                let grouped = pending.narrow(1, 0, n_grouped)?.contiguous()?;
                let new_embeds = model.adapter.forward(&grouped)?.to_dtype(model.dtype)?;
                let embeds = match self.embeds.take() {
                    Some(prev) => Tensor::cat(&[&prev, &new_embeds], 1)?,
                    None => new_embeds,
                };
                self.n_embeds = embeds.dim(1)?;
                *model
                    .audio_embeds_cache
                    .lock()
                    .expect("audio_embeds_cache lock") = Some(embeds.clone());
                self.embeds = Some(embeds);
            }
            if n_grouped < n {
                self.enc_pending = Some(pending.narrow(1, n_grouped, n - n_grouped)?.contiguous()?);
            }
        }

        Ok(())
    }

    /// Run the decoder as far as the available audio embeddings allow.
    ///
    /// A forward at position `p` reads audio embedding `p`, so during streaming
    /// (`finish_cap == None`) it may run once `p < n_embeds`; every such forward
    /// is then identical to the one whole-clip generation performs at the same
    /// position. At finish, generation instead drains to the whole-clip cap
    /// (positions past the audio length run text-only, as in whole-clip).
    fn advance_decoder(&mut self, model: &VoxtralModel, finish_cap: Option<usize>) -> Result<()> {
        if self.hit_eos {
            return Ok(());
        }

        if !self.prompt_fed {
            // The 39-token prompt needs audio embeddings for all its positions to
            // match whole-clip prefill; at finish it runs unconditionally (as does
            // whole-clip prefill, whatever the clip length).
            if self.n_embeds < N_PROMPT_TOKENS && finish_cap.is_none() {
                return Ok(());
            }
            let mut ids = Vec::with_capacity(N_PROMPT_TOKENS);
            ids.push(BOS_TOKEN_ID);
            ids.extend(std::iter::repeat_n(
                STREAMING_PAD_TOKEN_ID,
                N_PROMPT_TOKENS - 1,
            ));
            let logits = forward_ids(model, &ids, 0)?;
            let tok = argmax_id(&logits)?;
            self.tokens_fed = N_PROMPT_TOKENS;
            self.prompt_fed = true;
            if self.eos_toks.contains(&tok) && finish_cap.is_some() {
                self.hit_eos = true;
                return Ok(());
            }
            // During streaming an EOS is an utterance boundary, not the end
            // of dictation: it stays in the autoregressive context (the
            // detokenizer skips special tokens, so it never becomes text)
            // and decoding continues, so speech after a pause is heard.
            self.gen_ids.push(tok);
        }

        loop {
            match finish_cap {
                Some(cap) => {
                    if self.gen_ids.len() >= cap {
                        break;
                    }
                }
                None => {
                    if self.tokens_fed >= self.n_embeds {
                        break;
                    }
                }
            }
            let last = *self
                .gen_ids
                .last()
                .expect("decode loop requires a previous token");
            let logits = forward_ids(model, &[last], self.tokens_fed)?;
            let tok = argmax_id(&logits)?;
            self.tokens_fed += 1;
            if self.eos_toks.contains(&tok) && finish_cap.is_some() {
                // The speaker has stopped for good: the drain ends at the
                // utterance end, exactly as whole-clip generation does.
                self.hit_eos = true;
                break;
            }
            // Streaming: an utterance boundary stays in context, produces no
            // text, and never ends the session (the owner's first real pause
            // must not end dictation).
            self.gen_ids.push(tok);
        }

        Ok(())
    }

    /// Decode all generated ids and return the not-yet-emitted suffix. Decoding
    /// from scratch each time keeps deltas correct across BPE merge boundaries.
    fn take_delta(&mut self) -> Result<String> {
        let full = self
            .tokenizer
            .decode(&self.gen_ids, true)
            .map_err(|e| anyhow!("detokenization failed: {e}"))
            .context("decoding streaming ASR output")?;
        let delta = match full.strip_prefix(self.emitted.as_str()) {
            Some(rest) => rest.to_string(),
            None => {
                // A rare re-decode changed already-emitted text; emit from the
                // longest common prefix (the caller cannot retract text).
                let common = self
                    .emitted
                    .char_indices()
                    .zip(full.chars())
                    .find(|((_, a), b)| a != b)
                    .map(|((idx, _), _)| idx)
                    .unwrap_or_else(|| self.emitted.len().min(full.len()));
                full[common..].to_string()
            }
        };
        self.emitted = full;
        Ok(delta)
    }
}
