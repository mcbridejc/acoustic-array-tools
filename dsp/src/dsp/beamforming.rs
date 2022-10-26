use crate::buffer::{SampleBuffer, Spectra};
use crate::fft::fftimpl;
use num_complex::Complex;
use embassy_futures::yield_now;
use ndarray::{Array2, ArrayView3, Array3};

const SPEED_OF_SOUND: f32 = 343.0;

pub trait BeamFormer {
    fn compute_power(
            &self,
            spectra: & dyn Spectra,
            power_out: &mut [f32],
            start_freq: f32,
            end_freq: f32);
}

/// BeamFormer implementation using heap for storing steering vectors. Parameters, such as the
/// number of mics, number of focal points, or size of NFFT can be configured at runtime
pub struct HeapBeamFormer {
    steering_vectors: Array3<Complex<f32>>,
    sample_freq: f32,
}

impl HeapBeamFormer {
    pub fn new(sample_freq: f32) -> Self {
        Self {
            steering_vectors: Array3::zeros((0,0,0)),
            sample_freq
        }
    }

    /// Setup the beamformer
    ///
    /// mics: An NCHAN x 3 array of microphone positions in cartesian (x, y, z) coords
    /// focal_points: An NFOCAL x 3 array of focal points at which to calculate power
    /// nfft: The number of samples in each FFT spectrum
    /// sample_freq: The sample frequency of input audio
    pub fn setup(&mut self,
        mics: &Array2<f32>,
        focal_points: &Array2<f32>,
        nfft: usize,
        sample_freq: f32
    ) {
        assert!(mics.dim().1 == 3);
        assert!(focal_points.dim().1 == 3);
        let nchan = mics.dim().0;
        let nfocal = focal_points.dim().0;

        self.steering_vectors = Array3::zeros((nchan, nfocal, nfft));
        for i in 0..nchan {
            for j in 0..nfocal {
                let m = mics.row(i);
                let fp = focal_points.row(j);
                let mut sum_sqr: f32 = 0.0;
                for dim in 0..3 {
                    let x = m[dim] - fp[dim];
                    sum_sqr += x * x;
                }
                let d = libm::sqrtf(sum_sqr);
                for k in 0..nfft {
                    // Center frequency of the FFT bin
                    let bin_freq = k as f32 * sample_freq / 2.0 / (nfft) as f32;
                    // phase shift at center frequency based on distance between source and mic
                    let angle = core::f32::consts::PI * 2.0f32 * bin_freq * d / SPEED_OF_SOUND;
                    self.steering_vectors[[i, j, k]] = Complex::from_polar(1.0, angle);
                }
            }
        }
        self.sample_freq = sample_freq;
    }
}

impl BeamFormer for HeapBeamFormer {
    fn compute_power(
            &self,
            spectra: & dyn Spectra,
            power_out: &mut [f32],
            start_freq: f32,
            end_freq: f32) {
        let (nchan, nfocal, nfft) = self.steering_vectors.dim();
        let s = spectra.as_array().expect("empty spectra in compute_power");
        // Provided spectra must have the expected number of channels and frequency bins
        assert!(s.dim().0 == nchan, "Spectra has wrong number of channels");
        assert!(s.dim().1 == nfft, "Spectra has wrong number of frequencies");
        assert!(power_out.len() == nfocal, "Power out has wrong size ({} provided; expected {}", power_out.len(), nfocal);

        let freq_bin_start = libm::floorf(2.0 * start_freq * nfft as f32 / self.sample_freq) as usize;
        let freq_bin_end = libm::ceilf(2.0 * end_freq * nfft as f32 / self.sample_freq) as usize;

        assert!(freq_bin_end >= freq_bin_start, "Invalid frequencies");

        for i in 0..nfocal {
            let mut power_sum: f32 = 0.0;
            for bin in freq_bin_start..(freq_bin_end + 1) {
                let mut complex_sum = Complex::<f32> { im: 0.0, re: 0.0 };
                for ch in 0..nchan {
                    complex_sum += self.steering_vectors[[ch, i, bin]] * s[[ch,bin]];
                }
                power_sum += complex_sum.norm();

            }
            power_out[i] = power_sum / (freq_bin_end - freq_bin_start + 1) as f32;
            power_out[i] = 20.0 * libm::log10f(power_out[i]);
        }
    }
}

/// BeamFormer implementation with fixed parameters known at compile time, and no heap usage
pub struct StaticBeamFormer<const NCHAN: usize, const NFOCAL: usize, const NFFT: usize> {
    // When generic const exprs goes stable, this could just be an array of length {NCHAN * NFOCAL *
    // NFFT}
    steering_vectors: [[[Complex<f32>; NFFT]; NFOCAL]; NCHAN],
    sample_freq: f32,
}

impl<const NCHAN: usize, const NFOCAL: usize, const NFFT: usize> StaticBeamFormer<NCHAN, NFOCAL, NFFT> {
    pub const fn new() -> Self {
        let bf = Self {
            steering_vectors: [[[Complex::<f32>{re: 0.0, im: 0.0}; NFFT]; NFOCAL]; NCHAN],
            sample_freq: 0.0,
        };

        bf
    }

    pub fn setup(
        &mut self,
        mics: [[f32; 3]; NCHAN],
        focal_points: [[f32; 3]; NFOCAL],
        sample_freq: f32
    )
    {
        for i in 0..NCHAN {
            for j in 0..NFOCAL {
                let m = mics[i];
                let fp = focal_points[j];
                let mut sum_sqr: f32 = 0.0;
                for dim in 0..3 {
                    let x = m[dim] - fp[dim];
                    sum_sqr += x * x;
                }
                let d = libm::sqrtf(sum_sqr);
                for k in 0..NFFT {
                    // Center frequency of the FFT bin
                    let bin_freq = k as f32 * sample_freq / 2.0 / (NFFT - 1) as f32;
                    // phase shift at center frequency based on distance between source and mic
                    let angle = core::f32::consts::PI * 2.0f32 * bin_freq * d / SPEED_OF_SOUND;
                    self.steering_vectors[i][j][k] = Complex::from_polar(1.0, angle);
                }
            }
        }
        self.sample_freq = sample_freq;
    }
}

impl<
    const NCHAN: usize,//let Spectra = Spectra::
    const NFOCAL: usize,
    const NFFT: usize
> BeamFormer for StaticBeamFormer<NCHAN, NFOCAL, NFFT>
{
    fn compute_power(
        &self,
        spectra: & dyn Spectra,
        power_out: &mut [f32],
        start_freq: f32,
        end_freq: f32
    ) {
        let freq_bin_start = libm::floorf(2.0 * start_freq * NFFT as f32 / self.sample_freq) as usize;
        let freq_bin_end = libm::ceilf(2.0 * end_freq * NFFT as f32 / self.sample_freq) as usize;

        assert!(freq_bin_end >= freq_bin_start);
        assert!(spectra.nfft() == NFFT);
        assert!(spectra.channels() == NCHAN);

        let s = spectra.as_array().expect("empty spectra in compute_power");
        for i in 0..NFOCAL {
            let mut power_sum: f32 = 0.0;
            for bin in freq_bin_start..(freq_bin_end + 1) {
                let mut complex_sum = Complex::<f32> { im: 0.0, re: 0.0 };
                for ch in 0..NCHAN {
                    complex_sum += self.steering_vectors[ch][i][bin] * s[[ch,bin]];
                }
                power_sum += complex_sum.norm();

            }
            power_out[i] = power_sum / (freq_bin_end - freq_bin_start) as f32;
            power_out[i] = 20.0 * libm::log10f(power_out[i]);
        }
    }
}

/// Compute the distance between two vectors as slices
fn slice_cartesian_dist(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    let mut sum_sqr = 0.0f32;
    for dim in 0..a.len() {
        let x = a[dim] - b[dim];
        sum_sqr += x * x;
    }
    libm::sqrtf(sum_sqr)
}


/// A Beamformer implementation that does less pre-computation to avoid storage of large steering
/// vectors
pub struct SmallMemBeamFormer<const NCHAN: usize, const NFOCAL: usize>
where [(); NCHAN - 1]:
{
    dist: [[f32; NFOCAL]; NCHAN - 1],
    sample_freq: f32,
}

impl<const NCHAN: usize, const NFOCAL: usize> SmallMemBeamFormer<NCHAN, NFOCAL>
where [(); NCHAN - 1]:
{
    pub const fn new() -> Self {
        Self { dist: [[0.0; NFOCAL]; NCHAN - 1], sample_freq: 0.0 }
    }

    /// Perform prep computations
    ///
    /// On SmallMemBeamFormer, distances from focal points to mics are pre-calculated
    pub fn setup(
        &mut self,
        mics: [[f32; 3]; NCHAN],
        focal_points: [[f32; 3]; NFOCAL],
        sample_freq: f32
    )
    {
        self.sample_freq = sample_freq;

        for j in 0..NFOCAL {
            // Distances are calculated relative to mic channel 0. This way we can skip the phase
            // adjustment calculation for that channel later.
            let d0 = slice_cartesian_dist(&mics[0], &focal_points[j]);
            for i in 1..NCHAN {
                self.dist[i-1][j] = slice_cartesian_dist(&mics[i], &focal_points[j]) - d0;
            }
        }
    }

    fn steering_value(&self, chan: usize, focal_point: usize, freq: f32) -> Complex<f32> {
        if chan == 0 {
            // No phase adjustment for channel 0; it's the reference channel
            Complex{ im: 0.0, re: 1.0 }
        } else {
            let dist = self.dist[chan-1][focal_point];
            // phase shift at center frequency based on distance between source and mic
            let angle = core::f32::consts::PI * 2.0f32 * freq * dist / SPEED_OF_SOUND;
            // TODO: The trig functions in from_polar are probably slow, and we can probably do with fairly low
            // accuracy here; should try out a LUT for this
            Complex::from_polar(1.0, angle)
        }
    }
}

impl<const NCHAN: usize, const NFOCAL: usize> BeamFormer for SmallMemBeamFormer<NCHAN, NFOCAL>
where [(); NCHAN - 1]:
{
    fn compute_power(
        &self,
        spectra: & dyn Spectra,
        power_out: &mut [f32],
        start_freq: f32,
        end_freq: f32
    ) {
        let nfft = spectra.nfft();
        let freq_bin_start = libm::floorf(2.0 * start_freq * nfft as f32 / self.sample_freq) as usize;
        let freq_bin_end = libm::ceilf(2.0 * end_freq * nfft as f32 / self.sample_freq) as usize;

        assert!(freq_bin_end >= freq_bin_start);
        assert!(freq_bin_end < nfft);
        assert!(spectra.channels() == NCHAN);

        let s = spectra.as_array().expect("empty spectra in compute_power");
        for i in 0..NFOCAL {
            let mut power_sum: f32 = 0.0;
            for bin in freq_bin_start..(freq_bin_end + 1) {
                let mut complex_sum = Complex::<f32> { im: 0.0, re: 0.0 };
                let bin_freq = bin as f32 * self.sample_freq / 2.0 / (nfft - 1) as f32;
                for ch in 0..NCHAN {
                    complex_sum += self.steering_value(ch, i, bin_freq) * s[[ch,bin]];
                }
                power_sum += complex_sum.norm();
            }
            power_out[i] = power_sum / (freq_bin_end - freq_bin_start + 1) as f32;
            power_out[i] = 20.0 * libm::log10f(power_out[i]);
        }
    }
}




