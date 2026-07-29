#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rubato::Resampler;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

use super::config::AudioEncodingArgs;

/// Number of silence tokens to left-pad audio (matches voxmlx reference).
pub(super) const N_LEFT_PAD_TOKENS: usize = 32;
/// Number of silence tokens to right-pad audio (matches voxmlx reference).
pub(super) const N_RIGHT_PAD_TOKENS: usize = 17;

/// Whisper-style mel spectrogram processor for Voxtral audio encoder.
pub struct VoxtralAudioProcessor {
    sampling_rate: u32,
    frame_rate: f32,
    num_mel_bins: usize,
    hop_length: usize,
    window_size: usize,
    global_log_mel_max: f32,
}

/// Per-frame mel computation machinery (Hann window, FFT plan, mel filterbank).
///
/// Both the whole-clip path ([`VoxtralAudioProcessor::process_audio`]) and the
/// streaming session compute every mel frame through [`MelFrameEngine::compute_frames`],
/// so the two paths are bit-identical for identical total sample streams.
pub(super) struct MelFrameEngine {
    hop: usize,
    n_fft: usize,
    num_mel_bins: usize,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
    fft: Arc<dyn Fft<f32>>,
    log_mel_floor: f32,
}

impl MelFrameEngine {
    /// Total number of mel frames a clip of `len` samples produces.
    ///
    /// Matches `torch.stft(center=True)` with the final frame dropped:
    /// reflection padding adds `n_fft/2` on each side, so
    /// `total = (len + 2*(n_fft/2) - n_fft)/hop + 1 - 1 = len/hop`.
    pub(super) fn total_frames(&self, len: usize) -> usize {
        len / self.hop
    }

    /// Number of leading frames computable from `len` samples without touching
    /// the right reflection pad (i.e. without knowing any future samples).
    ///
    /// Frame `f` reads samples `f*hop - n_fft/2 .. f*hop + n_fft/2`, so it is
    /// final once `f*hop + n_fft/2 <= len`.
    pub(super) fn ready_frames(&self, len: usize) -> usize {
        let pad = self.n_fft / 2;
        if len < pad {
            0
        } else {
            let ready = (len - pad) / self.hop + 1;
            debug_assert!(ready <= self.total_frames(len));
            ready
        }
    }

    /// Compute mel frames `start..end` over `samples`, returned as a flat
    /// `(end - start) * num_mel_bins` vector.
    ///
    /// The clip start is reflection-padded (matching `torch.stft(center=True)`).
    /// If `reflect_right` is set, reads past the end reflect at the clip end as
    /// well (only valid once the clip is complete); otherwise every requested
    /// frame must satisfy `frame*hop + n_fft/2 <= samples.len()`.
    pub(super) fn compute_frames(
        &self,
        samples: &[f32],
        start: usize,
        end: usize,
        reflect_right: bool,
    ) -> Vec<f32> {
        let n_fft = self.n_fft;
        let pad = n_fft / 2;
        let n_freqs = n_fft / 2 + 1;
        let len = samples.len() as isize;

        // Identical indexing to building the reflection-padded buffer explicitly:
        // padded[i] = samples[(pad - i).min(len - 1)]        for i < pad
        // padded[pad + len + j] = samples[len - 2 - j]        (saturating) for the right pad
        let sample_at = |s: isize| -> f32 {
            if s < 0 {
                samples[((-s) as usize).min(samples.len() - 1)]
            } else if s < len {
                samples[s as usize]
            } else {
                debug_assert!(reflect_right, "frame reads past available samples");
                samples[samples.len().saturating_sub(2 + (s - len) as usize)]
            }
        };

        let mut out = Vec::with_capacity((end - start) * self.num_mel_bins);
        let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_fft];

        for frame_idx in start..end {
            let start_s = frame_idx as isize * self.hop as isize - pad as isize;
            for (i, (b, &w)) in buf.iter_mut().zip(self.window.iter()).enumerate() {
                *b = Complex32::new(sample_at(start_s + i as isize) * w, 0.0);
            }

            self.fft.process(&mut buf);

            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

            for filter in self.mel_filters.iter() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < power.len() {
                        sum += power[freq_idx] * coeff;
                    }
                }
                let log_val = sum.max(1e-10).log10();
                let clamped = log_val.max(self.log_mel_floor);
                out.push((clamped + 4.0) / 4.0);
            }
        }

        out
    }

    pub(super) fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }
}

impl VoxtralAudioProcessor {
    pub fn new(cfg: &AudioEncodingArgs) -> Self {
        Self {
            sampling_rate: cfg.sampling_rate,
            frame_rate: cfg.frame_rate as f32,
            num_mel_bins: cfg.num_mel_bins,
            hop_length: cfg.hop_length,
            window_size: cfg.window_size,
            global_log_mel_max: cfg.global_log_mel_max as f32,
        }
    }

    pub fn new_from_processor(other: &Self) -> Self {
        Self {
            sampling_rate: other.sampling_rate,
            frame_rate: other.frame_rate,
            num_mel_bins: other.num_mel_bins,
            hop_length: other.hop_length,
            window_size: other.window_size,
            global_log_mel_max: other.global_log_mel_max,
        }
    }

    /// Number of samples per streaming token (sampling_rate / frame_rate).
    pub(super) fn samples_per_token(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate) as usize
    }

    /// Build the per-frame mel engine for this processor's parameters.
    pub(super) fn frame_engine(&self) -> Result<MelFrameEngine> {
        let n_fft = self.window_size;

        // Hann window (periodic: w[n] = 0.5*(1 - cos(2*pi*n/N)))
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();

        let mel_filters = self.create_mel_filterbank(n_fft)?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Ok(MelFrameEngine {
            hop: self.hop_length,
            n_fft,
            num_mel_bins: self.num_mel_bins,
            window,
            mel_filters,
            fft,
            log_mel_floor: self.global_log_mel_max - 8.0,
        })
    }

    /// Process audio input into a mel spectrogram tensor.
    /// Left-pads with 32 tokens of silence and right-pads with 17 tokens of silence
    /// to match the reference implementation.
    /// Returns [1, T, num_mel_bins] tensor.
    pub fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();

        // Resample if necessary
        let samples = if audio.sample_rate != self.sampling_rate {
            self.resample(&mono, audio.sample_rate, self.sampling_rate)?
        } else {
            mono
        };

        // Pad audio with silence: left_pad + audio + right_pad
        let spt = self.samples_per_token();
        let left_pad = N_LEFT_PAD_TOKENS * spt;
        let right_pad = N_RIGHT_PAD_TOKENS * spt;
        let mut padded = vec![0.0f32; left_pad + samples.len() + right_pad];
        padded[left_pad..left_pad + samples.len()].copy_from_slice(&samples);

        let engine = self.frame_engine()?;
        let num_frames = if padded.is_empty() {
            0
        } else {
            engine.total_frames(padded.len())
        };
        if num_frames == 0 {
            anyhow::bail!("Audio too short to produce mel frames");
        }
        let data = engine.compute_frames(&padded, 0, num_frames, true);

        let tensor = Tensor::from_vec(data, (1, num_frames, self.num_mel_bins), device)?;
        Ok(tensor)
    }

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }
        let sinc = rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let mut resampler = rubato::SincFixedIn::<f32>::new(
            to_rate as f64 / from_rate as f64,
            2.0,
            sinc,
            samples.len(),
            1,
        )?;
        let result = resampler.process(&[samples.to_vec()], None)?;
        Ok(result[0].clone())
    }

    /// Slaney mel scale: Hz to mel.
    fn hertz_to_mel(freq: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 27.0 / 1.856_298; // 27.0 / ln(6.4)
        if freq >= MIN_LOG_HERTZ {
            MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
        } else {
            3.0 * freq / 200.0
        }
    }

    /// Slaney mel scale: mel to Hz.
    fn mel_to_hertz(mel: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 1.856_298 / 27.0; // ln(6.4) / 27.0
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HERTZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
        } else {
            200.0 * mel / 3.0
        }
    }

    /// Create Slaney-style mel filterbank matching `mistral_common.audio.mel_filter_bank`.
    /// Returns `[n_mels][n_freqs]` with Slaney energy normalization.
    fn create_mel_filterbank(&self, n_fft: usize) -> Result<Vec<Vec<f32>>> {
        let n_freqs = n_fft / 2 + 1;
        let sr = self.sampling_rate as f32;
        let n_mels = self.num_mel_bins;

        // FFT bin frequencies: linspace(0, sr/2, n_freqs)
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
            .collect();

        // Mel filter center frequencies (n_mels + 2 points)
        let mel_min = Self::hertz_to_mel(0.0);
        let mel_max = Self::hertz_to_mel(sr / 2.0);
        let filter_freqs: Vec<f32> = (0..n_mels + 2)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
                Self::mel_to_hertz(mel)
            })
            .collect();

        // Differences between adjacent filter frequencies
        let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

        // Triangular filterbank (matching _create_triangular_filter_bank)
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];
        for m in 0..n_mels {
            for (j, &fft_f) in fft_freqs.iter().enumerate() {
                let slope_left = fft_f - filter_freqs[m];
                let slope_right = filter_freqs[m + 2] - fft_f;
                let down = slope_left / filter_diff[m]; // rising slope
                let up = slope_right / filter_diff[m + 1]; // falling slope
                filterbank[m][j] = 0.0f32.max(down.min(up));
            }
        }

        // Slaney energy normalization: constant energy per channel
        for m in 0..n_mels {
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            for val in filterbank[m].iter_mut().take(n_freqs) {
                *val *= enorm;
            }
        }

        Ok(filterbank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> VoxtralAudioProcessor {
        VoxtralAudioProcessor::new(&AudioEncodingArgs {
            sampling_rate: 16000,
            frame_rate: 12.5,
            num_mel_bins: 128,
            hop_length: 160,
            window_size: 400,
            global_log_mel_max: 1.5,
        })
    }

    /// Deterministic pseudo-speech signal with a length that is not a multiple
    /// of the hop, the chunk size, or the streaming token size.
    fn synth_samples(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 733.0 * t + 0.5).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 2917.0 * t).sin()
            })
            .collect()
    }

    /// Incremental frame computation (chunked feed + finish) must be bit-identical
    /// to the whole-clip computation over the same padded sample stream.
    #[test]
    fn incremental_mel_matches_whole_clip() {
        let proc = processor();
        let engine = proc.frame_engine().unwrap();
        let spt = proc.samples_per_token();

        let speech = synth_samples(21931);

        // Whole-clip reference: left pad + speech + right pad, all frames at once.
        let mut whole = vec![0.0f32; N_LEFT_PAD_TOKENS * spt];
        whole.extend_from_slice(&speech);
        whole.extend(std::iter::repeat_n(0.0f32, N_RIGHT_PAD_TOKENS * spt));
        let total = engine.total_frames(whole.len());
        let reference = engine.compute_frames(&whole, 0, total, true);

        // Streaming: start from the left pad, feed odd-sized chunks, compute only
        // frames that need no right reflection; at finish, append the right pad and
        // compute the remainder with right reflection enabled.
        let mut buf = vec![0.0f32; N_LEFT_PAD_TOKENS * spt];
        let mut done = 0usize;
        let mut streamed: Vec<f32> = Vec::new();
        for chunk in speech.chunks(1237) {
            buf.extend_from_slice(chunk);
            let ready = engine.ready_frames(buf.len());
            if ready > done {
                streamed.extend(engine.compute_frames(&buf, done, ready, false));
                done = ready;
            }
        }
        buf.extend(std::iter::repeat_n(0.0f32, N_RIGHT_PAD_TOKENS * spt));
        let final_total = engine.total_frames(buf.len());
        assert_eq!(final_total, total);
        streamed.extend(engine.compute_frames(&buf, done, final_total, true));

        assert_eq!(streamed.len(), reference.len());
        for (i, (a, b)) in streamed.iter().zip(reference.iter()).enumerate() {
            assert!(a == b, "mel value diverged at flat index {i}: {a} vs {b}");
        }
    }
}
