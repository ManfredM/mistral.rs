//! Whole-clip mono resampling shared by the audio processors.
//!
//! rubato 1.0 moved buffers to `audioadapter` and gave its sinc resampler a
//! real start-up delay. `process_all` resamples the whole clip in chunks and
//! trims that delay, so the output starts at the first input sample, as the
//! 0.16 `SincFixedIn` sized to the clip did, and it no longer loses the last
//! few output frames.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// Input frames the resampler consumes per pass inside `process_all`.
const CHUNK: usize = 1024;

/// Resample mono `samples` from `from_rate` to `to_rate` with a 256-tap
/// Blackman-Harris sinc filter (cutoff 0.95), the filter these processors
/// have always used. Returns `ceil(len * to_rate / from_rate)` frames.
pub(crate) fn resample_mono(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> anyhow::Result<Vec<f32>> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let parameters = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(
        f64::from(to_rate) / f64::from(from_rate),
        1.0,
        &parameters,
        CHUNK,
        1,
        FixedAsync::Input,
    )?;
    let input = InterleavedSlice::new(samples, 1, samples.len())?;
    Ok(resampler
        .process_all(&input, samples.len(), None)?
        .take_data())
}

#[cfg(test)]
mod tests {
    use super::resample_mono;

    /// A tone resampled 48 kHz -> 16 kHz keeps its length, starts on time
    /// (within one output frame: no start-up delay) and keeps its shape.
    #[test]
    fn resample_mono_keeps_length_and_phase() {
        let tone = |t: f64| (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        let input: Vec<f32> = (0..48_000)
            .map(|i| tone(i as f64 / 48_000.0) as f32)
            .collect();
        let output = resample_mono(&input, 48_000, 16_000).unwrap();
        assert_eq!(output.len(), 16_000);
        // The worst deviation from the exact tone shifted by `shift` frames;
        // the filter keeps it near -50 dB (0.0027 measured).
        let worst_at = |shift: f64| {
            (1_000..15_000)
                .map(|n| (f64::from(output[n]) - tone((n as f64 - shift) / 16_000.0)).abs())
                .fold(0.0, f64::max)
        };
        let (shift, worst) = (-20..=20)
            .map(|step| f64::from(step) / 20.0)
            .map(|shift| (shift, worst_at(shift)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            worst < 5e-3,
            "best alignment {shift} frames still deviates by {worst}"
        );
    }

    #[test]
    fn resample_mono_passes_equal_rates_and_empty_input_through() {
        assert_eq!(
            resample_mono(&[0.5, -0.5], 16_000, 16_000).unwrap(),
            vec![0.5, -0.5]
        );
        assert!(resample_mono(&[], 48_000, 16_000).unwrap().is_empty());
    }
}
