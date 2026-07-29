#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    pipeline::{KvCache, NormalCache},
};

use super::config::WhisperEncoderArgs;

pub(super) struct EncoderAttention {
    pub(super) wq: Arc<dyn QuantMethod>,
    pub(super) wk: Arc<dyn QuantMethod>,
    pub(super) wv: Arc<dyn QuantMethod>,
    pub(super) wo: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl EncoderAttention {
    fn new(
        cfg: &WhisperEncoderArgs,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let num_heads = cfg.n_heads;
        let num_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let use_bias = cfg.use_biases;

        // Per-linear bias flags matching actual weight structure:
        // wq, wv, wo have bias; wk does NOT
        let wq =
            mistralrs_quant::linear_b(dim, num_heads * head_dim, use_bias, &None, vb.pp("wq"))?;
        let wk =
            mistralrs_quant::linear_b(dim, num_kv_heads * head_dim, false, &None, vb.pp("wk"))?;
        let wv =
            mistralrs_quant::linear_b(dim, num_kv_heads * head_dim, use_bias, &None, vb.pp("wv"))?;
        let wo =
            mistralrs_quant::linear_b(num_heads * head_dim, dim, use_bias, &None, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, positions)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // The mask is built from predicted cache accounting; if it ever
        // disagrees with what the cache actually yielded, fail with the
        // numbers instead of a bare broadcast error deep in attention.
        if let AttentionMask::Custom(m) = attention_mask {
            let k_len = k.dim(2)?;
            let m_len = m.dim(candle_core::D::Minus1)?;
            if m_len != k_len {
                candle_core::bail!(
                    "voxtral encoder mask width {m_len} != post-append key length {k_len} \
                     (q_len {q_len}): sliding-window accounting bug"
                );
            }
        }

        let attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            None, // no flash params for encoder (causal via mask)
            &self.sdpa_params,
        )?;

        let attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.wo.forward(&attn_output)
    }
}

pub(super) struct EncoderMlp {
    pub(super) w1: Arc<dyn QuantMethod>, // gate
    pub(super) w2: Arc<dyn QuantMethod>, // down
    pub(super) w3: Arc<dyn QuantMethod>, // up
}

impl EncoderMlp {
    fn new(cfg: &WhisperEncoderArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let use_bias = cfg.use_biases;

        // Per-linear bias flags: only w2 has bias; w1, w3 do not
        let w1 = mistralrs_quant::linear_b(dim, hidden_dim, false, &None, vb.pp("w1"))?;
        let w2 = mistralrs_quant::linear_b(hidden_dim, dim, use_bias, &None, vb.pp("w2"))?;
        let w3 = mistralrs_quant::linear_b(dim, hidden_dim, false, &None, vb.pp("w3"))?;

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(w1(x)) * w3(x), then w2
        let gate = self.w1.forward(xs)?;
        let up = self.w3.forward(xs)?;
        let xs = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.w2.forward(&xs)
    }
}

pub(super) struct EncoderLayer {
    pub(super) attention: EncoderAttention,
    pub(super) feed_forward: EncoderMlp,
    pub(super) attention_norm: RmsNorm,
    pub(super) ffn_norm: RmsNorm,
}

impl EncoderLayer {
    fn new(
        cfg: &WhisperEncoderArgs,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let attention = EncoderAttention::new(cfg, rotary_emb, vb.pp("attention"))?;
        let feed_forward = EncoderMlp::new(cfg, vb.pp("feed_forward"))?;
        let attention_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs = self
            .attention
            .forward(&xs, attention_mask, positions, kv_cache)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        residual + xs
    }
}

/// Carried state for the incremental streaming convolution
/// ([`VoxtralEncoder::convolve_incremental`]).
///
/// Both convolutions are causal, so a chunk's output frames only need a fixed
/// amount of left context: conv1 (kernel 3, stride 1, left pad 2) needs the
/// previous 2 mel frames, and conv2 (kernel 3, stride 2, left pad 1) needs the
/// unconsumed tail of the conv1 output stream. Carrying exactly that context
/// makes the per-chunk cost proportional to the chunk, not the session, while
/// staying bit-identical to convolving the whole history at once.
pub struct StreamingConvState {
    /// The last 2 raw input frames fed to conv1, `[1, in_ch, 2]` (F32).
    /// `None` before the first chunk: the causal left pad of 2 zero frames.
    conv1_tail: Option<Tensor>,
    /// Conv1 output frames (post-GELU) not yet consumed by conv2,
    /// `[1, ch, 1..=2]` (F32). `None` before the first chunk: the causal left
    /// pad of 1 zero frame. The buffer's first frame always sits at an even
    /// index of the padded conv1 output stream, so conv2's stride-2 grid over
    /// the buffer lands exactly where the whole-clip computation puts it.
    conv2_buf: Option<Tensor>,
}

impl StreamingConvState {
    pub fn new() -> Self {
        Self {
            conv1_tail: None,
            conv2_buf: None,
        }
    }

    /// Push `new` input frames (`[1, in_ch, m]`, F32) through the two causal
    /// convolutions, returning every output frame that is now final
    /// (`[1, out_ch, n]`, post-GELU, possibly `n == 0`) and carrying the
    /// remainders. Feeding a stream in chunks yields, concatenated, exactly
    /// the frames the one-shot pad-and-convolve computation yields.
    fn advance(
        &mut self,
        conv1: &candle_nn::Conv1d,
        conv2: &candle_nn::Conv1d,
        new: &Tensor,
    ) -> Result<Option<Tensor>> {
        if new.dim(2)? == 0 {
            return Ok(None);
        }

        // Conv1: [carried 2-frame tail | new] -> exactly `m` final frames.
        let c1_in = match self.conv1_tail.take() {
            Some(tail) => Tensor::cat(&[&tail, new], 2)?,
            None => new.pad_with_zeros(2, 2, 0)?,
        };
        let w = c1_in.dim(2)?;
        self.conv1_tail = Some(c1_in.narrow(2, w - 2, 2)?.contiguous()?);
        let c1_out = conv1.forward(&c1_in.contiguous()?)?.gelu_erf()?;

        // Conv2: [carried tail | conv1 frames]; a window is final once all 3
        // of its inputs exist, and consuming 2 per output preserves the
        // stride-2 grid alignment across chunk boundaries.
        let buf = match self.conv2_buf.take() {
            Some(prev) => Tensor::cat(&[&prev, &c1_out], 2)?,
            None => c1_out.pad_with_zeros(2, 1, 0)?,
        };
        let l = buf.dim(2)?;
        if l < 3 {
            self.conv2_buf = Some(buf);
            return Ok(None);
        }
        let n_out = (l - 3) / 2 + 1;
        let out = conv2.forward(&buf.contiguous()?)?.gelu_erf()?;
        self.conv2_buf = Some(buf.narrow(2, 2 * n_out, l - 2 * n_out)?.contiguous()?);
        Ok(Some(out))
    }
}

impl Default for StreamingConvState {
    fn default() -> Self {
        Self::new()
    }
}

/// Causal Whisper-based audio encoder for Voxtral.
///
/// Unlike standard Whisper, this uses:
/// - Two Conv1d layers to project mel features to encoder dim
/// - Causal attention with sliding window (750 frames)
/// - RoPE positional embeddings
/// - SwiGLU FFN
/// - RMSNorm
pub struct VoxtralEncoder {
    pub(super) conv1: candle_nn::Conv1d,
    pub(super) conv2: candle_nn::Conv1d,
    pub(super) layers: Vec<EncoderLayer>,
    pub(super) norm: RmsNorm,
    cache: Arc<Mutex<NormalCache>>,
    #[allow(dead_code)]
    num_heads: usize,
    sliding_window: Option<usize>,
    n_layers: usize,
    /// Model dtype (e.g. BF16) for the transformer layers.
    /// Conv1d weights are stored as F32 for CUDA compatibility.
    model_dtype: DType,
}

impl VoxtralEncoder {
    pub fn new(cfg: &WhisperEncoderArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let n_mels = cfg.audio_encoding_args.num_mel_bins;

        // Conv1d weights stored as F32 (CUDA Conv1d does not support BF16).
        // Causal padding: left-pad by (kernel_size - 1) * dilation, padding=0 in Conv1d.
        let vb_c1 = vb.pp("conv_layers").pp("0").pp("conv");
        let conv1 = candle_nn::Conv1d::new(
            vb_c1
                .get((cfg.dim, n_mels, 3), "weight")?
                .to_dtype(DType::F32)?,
            Some(vb_c1.get(cfg.dim, "bias")?.to_dtype(DType::F32)?),
            candle_nn::Conv1dConfig {
                padding: 0,
                stride: 1,
                ..Default::default()
            },
        );
        let vb_c2 = vb.pp("conv_layers").pp("1").pp("conv");
        let conv2 = candle_nn::Conv1d::new(
            vb_c2
                .get((cfg.dim, cfg.dim, 3), "weight")?
                .to_dtype(DType::F32)?,
            Some(vb_c2.get(cfg.dim, "bias")?.to_dtype(DType::F32)?),
            candle_nn::Conv1dConfig {
                padding: 0,
                stride: 2,
                ..Default::default()
            },
        );

        // Create RoPE embeddings for encoder
        let mut ropes = HashMap::new();
        ropes.insert(
            device.location(),
            Arc::new(RotaryEmbedding::new(
                cfg.rope_theta as f32,
                cfg.head_dim,
                1_000_000, // large max_position for encoder
                &device,
                false, // !is_gptx: consolidated.safetensors stores Q/K in interleaved layout
                dtype,
            )?),
        );

        let vb_layers = vb.pp("transformer").pp("layers");
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            layers.push(EncoderLayer::new(cfg, rotary_emb, vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("transformer").pp("norm"))?;

        Ok(Self {
            conv1,
            conv2,
            layers,
            norm,
            cache: NormalCache::new_sliding(cfg.n_layers, 1_000_000, cfg.sliding_window),
            num_heads: cfg.n_heads,
            sliding_window: cfg.sliding_window,
            n_layers: cfg.n_layers,
            model_dtype: dtype,
        })
    }

    /// Forward pass through the encoder.
    /// Input: mel features [B, T, mel_bins]
    /// Output: [B, T/2, dim]
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let conv = self.convolve(xs)?;
        self.encode_new(&conv)
    }

    /// The two causal convolutions, without the transformer.
    /// Input: mel features [B, T, mel_bins]; output: conv frames [B, T/2, dim]
    /// in the model dtype, ready for [`Self::encode_new`].
    ///
    /// Both convolutions are causal (left-padded), so recomputing them over the
    /// full mel history always reproduces earlier frames exactly: frame `j` of
    /// the output only depends on mel frames `..=2*j+1`.
    pub fn convolve(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?;

        // Transpose [B, T, mel] -> [B, mel, T] for Conv1d
        let xs = xs.transpose(1, 2)?;

        // Causal Conv1: left-pad by 2, then conv(kernel=3, stride=1, padding=0)
        let xs = xs.pad_with_zeros(2, 2, 0)?;
        let xs = self.conv1.forward(&xs)?.gelu_erf()?;

        // Causal Conv2: left-pad by 1, then conv(kernel=3, stride=2, padding=0)
        // HF VoxtralRealtimeCausalConv1d stores left_pad=1 for this layer.
        let xs = xs.pad_with_zeros(2, 1, 0)?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;

        // Transpose back [B, dim, T/2] -> [B, T/2, dim]
        let xs = xs.transpose(1, 2)?.contiguous()?;
        // Cast from F32 to model dtype for transformer layers
        xs.to_dtype(self.model_dtype)
    }

    /// Incremental counterpart of [`Self::convolve`]: push only the newly
    /// final mel frames through both convolutions, carrying the bounded left
    /// context in `state`. Per-call cost is proportional to the chunk, not the
    /// mel history, and the concatenated outputs are identical to
    /// [`Self::convolve`] over the whole history.
    /// Input: new mel frames [B, m, mel_bins]; output: the conv frames that
    /// are now final, [B, n, dim] in the model dtype (possibly none).
    pub fn convolve_incremental(
        &self,
        mel_new: &Tensor,
        state: &mut StreamingConvState,
    ) -> Result<Option<Tensor>> {
        let xs = mel_new.to_dtype(DType::F32)?.transpose(1, 2)?;
        match state.advance(&self.conv1, &self.conv2, &xs)? {
            Some(out) => Ok(Some(
                out.transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(self.model_dtype)?,
            )),
            None => Ok(None),
        }
    }

    /// Run the transformer over conv frames that have not been encoded yet,
    /// appending to the persistent KV cache. RoPE positions continue from the
    /// number of frames already in the cache, so feeding a clip in chunks is
    /// equivalent to feeding it at once (the attention is causal).
    /// Input: conv frames [B, S, dim]; output: [B, S, dim].
    pub fn encode_new(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _dim) = xs.dims3()?;

        let mut cache = self.cache.lock().expect("Encoder cache lock poisoned");
        // `RotatingCache::current_seq_len` is the TOTAL number of frames ever
        // appended — it keeps growing past the sliding window and is never
        // truncated by eviction — so it is the absolute stream position of the
        // first new frame. That is exactly what RoPE needs: the same absolute,
        // monotonic positions the text decoders pass via `seqlen_offsets`.
        let past = cache.0[0].current_seq_len();

        // Per-token RoPE positions: past..past+seq_len for each batch row.
        let mut pos = Vec::with_capacity(b_sz * seq_len);
        for _ in 0..b_sz {
            pos.extend((past..past + seq_len).map(|p| p as u32));
        }
        let positions = Tensor::from_vec(pos, b_sz * seq_len, xs.device())?;

        let attention_mask = if seq_len == 1 {
            // One query after an append sees exactly the retained window
            // (the newest `min(past + 1, sw)` keys), which is precisely its
            // visible set under the causal sliding window: no mask needed.
            AttentionMask::None
        } else if let Some(sw) = self.sliding_window {
            // The mask must be as wide as the keys `KvCache::append` actually
            // returns for this call. Once the window has saturated, the
            // rotating cache's multi-token append returns
            // `retained-before-append + new = min(past, sw) + seq_len` keys —
            // NOT `past + seq_len`: frames older than the window are gone.
            // `CausalMasker::make_swa_mask` assumes key column `j` sits at
            // absolute position `j` (nothing evicted), so past saturation it
            // is both too wide and mis-banded; build the band over the keys
            // actually present instead. Below saturation
            // (`retained_past == past`) this reduces to exactly that helper's
            // mask, so whole-clip and early streaming behavior are unchanged.
            let retained_past = past.min(sw);
            let k_len = retained_past + seq_len;
            // Absolute position of key column 0.
            let oldest_k_pos = past - retained_past;
            let mut mask = Vec::with_capacity(seq_len * k_len);
            for i in 0..seq_len {
                let q_pos = past + i;
                for j in 0..k_len {
                    let k_pos = oldest_k_pos + j;
                    // Causal, with HF's exclusive lower bound
                    // (`kv_idx > q_idx - sliding_window`): the token plus its
                    // visible history total exactly `sw` positions.
                    let visible = k_pos <= q_pos && q_pos - k_pos < sw;
                    mask.push(if visible { 0f32 } else { f32::NEG_INFINITY });
                }
            }
            AttentionMask::Custom(
                Tensor::from_slice(&mask, (seq_len, k_len), xs.device())?.to_dtype(xs.dtype())?,
            )
        } else {
            // No sliding window: nothing is ever evicted, the generic helper's
            // key-position assumption holds.
            let dummy_toks = Tensor::zeros((b_sz, seq_len), DType::U32, xs.device())?;
            CausalMasker.make_causal_mask(
                &dummy_toks,
                &cache.0 as &dyn PastKvLenCache,
                xs.dtype(),
                &CausalMaskConfig::default(),
            )?
        };

        let mut hidden = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &attention_mask, &positions, &mut cache.0[i])?;
        }

        self.norm.forward(&hidden)
    }

    /// Reset the encoder KV cache (call between different audio inputs).
    pub fn reset_cache(&self) {
        let fresh = NormalCache::new_sliding(self.n_layers, 1_000_000, self.sliding_window);
        let inner = fresh.lock().expect("New cache lock poisoned").clone();
        *self.cache.lock().expect("Encoder cache lock poisoned") = inner;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Deterministic pseudo-random conv weights (no model download needed).
    fn test_conv(
        out_ch: usize,
        in_ch: usize,
        stride: usize,
        seed: u32,
        device: &Device,
    ) -> candle_nn::Conv1d {
        let n = out_ch * in_ch * 3;
        let mut state = seed;
        let mut next = || {
            // xorshift32; values in roughly [-0.5, 0.5]
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) - 0.5
        };
        let weight: Vec<f32> = (0..n).map(|_| next()).collect();
        let bias: Vec<f32> = (0..out_ch).map(|_| next()).collect();
        candle_nn::Conv1d::new(
            Tensor::from_vec(weight, (out_ch, in_ch, 3), device).unwrap(),
            Some(Tensor::from_vec(bias, out_ch, device).unwrap()),
            candle_nn::Conv1dConfig {
                padding: 0,
                stride,
                ..Default::default()
            },
        )
    }

    /// Chunked incremental convolution must reproduce the one-shot
    /// pad-and-convolve pipeline over the same frame stream, for chunk sizes
    /// that hit every stride-parity and sub-kernel-carry case.
    ///
    /// The comparison uses a tight tolerance rather than bit equality: candle
    /// lowers conv1d to im2col + gemm, whose accumulation order (and thus the
    /// last ULP) legitimately varies with the call's width. Any alignment,
    /// parity, or carry bug produces grossly different values, which this
    /// still catches; end-to-end exactness on the real checkpoint is what the
    /// streaming-vs-whole-clip transcript gate verifies.
    #[test]
    fn incremental_convolution_matches_whole_clip() {
        let device = Device::Cpu;
        let in_ch = 8;
        let mid_ch = 6;
        let out_ch = 6;
        let conv1 = test_conv(mid_ch, in_ch, 1, 0x1234_5678, &device);
        let conv2 = test_conv(out_ch, mid_ch, 2, 0x9abc_def1, &device);

        let t_total = 137usize;
        let mut vals = (0..in_ch * t_total).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0);
        let input = Tensor::from_vec(
            (0..in_ch * t_total).map(|_| vals.next().unwrap()).collect(),
            (1, in_ch, t_total),
            &device,
        )
        .unwrap();

        // Whole-clip reference: exactly VoxtralEncoder::convolve's pipeline.
        let whole = {
            let xs = input.pad_with_zeros(2, 2, 0).unwrap();
            let xs = conv1.forward(&xs).unwrap().gelu_erf().unwrap();
            let xs = xs.pad_with_zeros(2, 1, 0).unwrap();
            conv2.forward(&xs).unwrap().gelu_erf().unwrap()
        };
        // Compare time-major so per-chunk outputs concatenate along time.
        let whole_flat = whole
            .transpose(1, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for chunk_sizes in [
            vec![1usize; t_total],
            vec![2; (t_total + 1) / 2],
            vec![3; (t_total + 2) / 3],
            vec![7, 1, 2, 5, 3, 1, 1, 4, 6, 2, 9, 1, 100],
        ] {
            let mut state = StreamingConvState::new();
            let mut got: Vec<f32> = Vec::new();
            let mut fed = 0usize;
            for &sz in &chunk_sizes {
                let take = sz.min(t_total - fed);
                if take == 0 {
                    break;
                }
                let chunk = input.narrow(2, fed, take).unwrap().contiguous().unwrap();
                fed += take;
                if let Some(out) = state.advance(&conv1, &conv2, &chunk).unwrap() {
                    got.extend(
                        out.transpose(1, 2)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap(),
                    );
                }
            }
            assert_eq!(fed, t_total);
            // Both paths emit a frame only once its full 3-input window
            // exists (padding 0), so the counts must match exactly.
            assert_eq!(
                got.len(),
                whole_flat.len(),
                "chunking {chunk_sizes:?}: frame count mismatch"
            );
            for (i, (a, b)) in got.iter().zip(whole_flat.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5f32.max(b.abs() * 1e-5),
                    "chunking {chunk_sizes:?}: conv output diverged at flat index {i}: {a} vs {b}"
                );
            }
        }
    }
}
