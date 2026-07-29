//! Incremental (streaming) ASR session for the Voxtral Realtime model.
//!
//! The whole-clip path processes a finished clip in one shot: pad the audio with
//! silence, compute the mel spectrogram, encode it, and autoregressively decode
//! one text token per audio-embedding position. Every stage of that pipeline is
//! causal, so the same result can be produced incrementally:
//!
//! - mel frames are final once their sample window is available
//!   ([`MelFrameEngine::ready_frames`]),
//! - conv frames only look left (causal convolutions, advanced incrementally
//!   with a carried context of a few frames — [`StreamingConvState`]),
//! - encoder frames only attend left (appending KV cache),
//! - adapter groups are independent per group of 4 encoder frames,
//! - a decoder forward at position `p` only needs audio embeddings `..=p`.
//!
//! Every per-feed cost is bounded by the feed, not the session: raw samples
//! are trimmed to the window future mel frames can read, the convolutions
//! carry O(1) context, audio embeddings are appended as segments (and evicted
//! once consumed) instead of re-concatenated, and detokenization decodes a
//! fixed tail window of ids. A session therefore keeps real-time pace
//! regardless of how long the speaker dictates.
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
use super::encoder::StreamingConvState;
use super::inputs_processor::{
    AUDIO_LENGTH_PER_TOK, BOS_TOKEN_ID, N_DELAY_TOKENS, STREAMING_PAD_TOKEN_ID,
};
use super::{AudioEmbedStore, VoxtralModel, VoxtralSpecificArgs};

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
    /// Retained raw 16 kHz mono samples: the bounded suffix of the global
    /// stream that future mel frames can still read (`samples[0]` is global
    /// sample index `samples_base`). The stream starts with the 32-token left
    /// pad of silence.
    samples: Vec<f32>,
    samples_base: usize,
    /// Number of mel frames computed so far.
    mel_done: usize,
    /// Carried context of the two causal convolutions (a few frames).
    conv_state: StreamingConvState,
    /// Encoder output frames awaiting a complete adapter group of 4.
    enc_pending: Option<Tensor>,
    /// Number of audio embeddings accumulated in the model's embed store.
    n_embeds: usize,
    downsample: usize,
    prompt_fed: bool,
    /// Number of decoder positions consumed (prompt + generated tokens fed back).
    tokens_fed: usize,
    /// Generated token ids (EOS excluded), in order.
    gen_ids: Vec<u32>,
    hit_eos: bool,
    /// Detokenization window: ids before `win_start` have left the window and
    /// their text is committed (immutable, already emitted); `emitted_win` is
    /// the text already emitted for ids at or after `win_start`.
    win_start: usize,
    emitted_win: String,
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
            samples_base: 0,
            mel_done: 0,
            conv_state: StreamingConvState::new(),
            enc_pending: None,
            n_embeds: 0,
            downsample,
            prompt_fed: false,
            tokens_fed: 0,
            gen_ids: Vec::new(),
            hit_eos: false,
            win_start: 0,
            emitted_win: String::new(),
        })
    }

    /// Append 16 kHz mono f32 samples as they arrive. Returns whatever text the
    /// model newly committed, in order (possibly empty).
    pub async fn feed(&mut self, pcm: &[f32]) -> Result<String> {
        let pipeline = Arc::clone(&self.pipeline);
        let guard = pipeline.lock().await;
        let model = voxtral_of(&*guard)?;

        self.samples.extend_from_slice(pcm);
        let new_mel = self.advance_mel(false);
        self.advance_encoder(model, new_mel)?;
        self.advance_decoder(model, None)?;
        // Embeddings at positions the decoder has consumed are never read
        // again; drop them so session memory stays bounded too.
        if let Some(store) = model
            .audio_embeds_cache
            .lock()
            .expect("audio_embeds_cache lock")
            .as_mut()
        {
            store.evict_below(self.tokens_fed);
        }
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
        let new_mel = self.advance_mel(true);
        self.advance_encoder(model, new_mel)?;

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

    /// Compute all mel frames that are final and return them (flattened
    /// `[frame][mel_bin]`). During streaming only frames whose full sample
    /// window exists are computed; at finish the remaining frames are computed
    /// with reflection at the (now final) clip end, exactly like the
    /// whole-clip path. Afterwards, raw samples no future frame can read are
    /// dropped, so the retained buffer stays bounded regardless of session
    /// length.
    fn advance_mel(&mut self, clip_complete: bool) -> Vec<f32> {
        let glen = self.samples_base + self.samples.len();
        let target = if clip_complete {
            self.mel_engine.total_frames(glen)
        } else {
            self.mel_engine.ready_frames(glen)
        };
        let new = if target > self.mel_done {
            let out = self.mel_engine.compute_frames(
                &self.samples,
                self.samples_base,
                self.mel_done,
                target,
                clip_complete,
            );
            self.mel_done = target;
            out
        } else {
            Vec::new()
        };

        let keep_from = self.mel_engine.retained_from(self.mel_done);
        if keep_from > self.samples_base {
            self.samples.drain(..keep_from - self.samples_base);
            self.samples_base = keep_from;
        }

        new
    }

    /// Push newly final mel frames through the convolutions (incrementally,
    /// with carried context), new conv frames through the encoder transformer,
    /// and complete adapter groups into the model's audio embedding store.
    fn advance_encoder(&mut self, model: &VoxtralModel, new_mel: Vec<f32>) -> Result<()> {
        // Conv frame j depends on mel frames ..=2j+1; the incremental
        // convolution emits exactly the frames whose full window now exists
        // and carries the remainders, so each feed costs O(new mel frames).
        if !new_mel.is_empty() {
            let device = MultimodalModel::device(model);
            let m = new_mel.len() / self.num_mel_bins;
            let mel = Tensor::from_vec(new_mel, (1, m, self.num_mel_bins), device)?;
            if let Some(conv_new) = model
                .encoder
                .convolve_incremental(&mel, &mut self.conv_state)?
            {
                let enc_out = model.encoder.encode_new(&conv_new)?;
                self.enc_pending = Some(match self.enc_pending.take() {
                    Some(prev) => Tensor::cat(&[&prev, &enc_out], 1)?,
                    None => enc_out,
                });
            }
        }

        // Adapter: complete groups of `downsample` encoder frames only; the
        // remainder waits (whole-clip drops a trailing partial group the same way).
        if let Some(pending) = self.enc_pending.take() {
            let n = pending.dim(1)?;
            let n_grouped = n - n % self.downsample;
            if n_grouped > 0 {
                let grouped = pending.narrow(1, 0, n_grouped)?.contiguous()?;
                let new_embeds = model.adapter.forward(&grouped)?.to_dtype(model.dtype)?;
                let n_new = new_embeds.dim(1)?;
                // Append as a fresh segment: O(new embeddings) per feed, no
                // re-copy of the session's embedding history.
                let mut store = model
                    .audio_embeds_cache
                    .lock()
                    .expect("audio_embeds_cache lock");
                match store.as_mut() {
                    Some(store) => store.push(new_embeds)?,
                    None => *store = Some(AudioEmbedStore::from_single(new_embeds)?),
                }
                self.n_embeds += n_new;
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

    /// How many generated ids the detokenization window keeps. Ids that leave
    /// the window have immutable, already-emitted text by construction, so
    /// each round decodes a bounded number of ids however long the session.
    const DELTA_WINDOW: usize = 64;

    /// Decode the tail window of generated ids and return the not-yet-emitted
    /// suffix. Re-decoding the window (rather than single tokens) keeps deltas
    /// correct across BPE merge boundaries and partial UTF-8 sequences at the
    /// tail, exactly as decoding from scratch did; the window start only ever
    /// moves at an id whose text boundary is verified clean, so text ahead of
    /// the window can never change.
    fn take_delta(&mut self) -> Result<String> {
        let window_text = self.decode_from(self.win_start)?;
        let delta = match window_text.strip_prefix(self.emitted_win.as_str()) {
            Some(rest) => rest.to_string(),
            None => {
                // A rare re-decode changed already-emitted text; emit from the
                // longest common prefix (the caller cannot retract text).
                let common = self
                    .emitted_win
                    .char_indices()
                    .zip(window_text.chars())
                    .find(|((_, a), b)| a != b)
                    .map(|((idx, _), _)| idx)
                    .unwrap_or_else(|| self.emitted_win.len().min(window_text.len()));
                window_text[common..].to_string()
            }
        };
        self.emitted_win = window_text;

        // Slide the window start forward once it exceeds the bound — but only
        // at an id where the boundary is clean: the shorter window must decode
        // to a suffix of the emitted window text, so the text moving out of
        // the window is exactly what was already emitted for those ids. If a
        // UTF-8 character spans the preferred boundary, a nearby id aligns (a
        // character is at most a few tokens); until one does, the window
        // simply stays a little longer.
        if self.gen_ids.len() - self.win_start > Self::DELTA_WINDOW {
            let target = self.gen_ids.len() - Self::DELTA_WINDOW;
            let lowest = target.saturating_sub(3).max(self.win_start + 1);
            for new_start in (lowest..=target).rev() {
                let rest = self.decode_from(new_start)?;
                if self.emitted_win.ends_with(rest.as_str()) {
                    self.emitted_win = rest;
                    self.win_start = new_start;
                    break;
                }
            }
        }

        Ok(delta)
    }

    /// Decode `gen_ids[from..]`, skipping special tokens (utterance-boundary
    /// EOS ids stay in context but never become text).
    fn decode_from(&self, from: usize) -> Result<String> {
        self.tokenizer
            .decode(&self.gen_ids[from..], true)
            .map_err(|e| anyhow!("detokenization failed: {e}"))
            .context("decoding streaming ASR output")
    }
}
