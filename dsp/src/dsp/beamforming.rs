use crate::buffer::{SampleBuffer, Spectra};
use num_complex::Complex;
use embassy_futures::yield_now;
use realfft::num_traits::Zero;
use ndarray::{Array2, ArrayView3, Array3};

const SPEED_OF_SOUND: f32 = 343.0;

#[cfg(feature="realfft")]
pub mod fftimpl {
    use realfft::{RealFftPlanner, RealToComplex};
    use std::sync::Arc;
    use super::Complex;

    /// Class for performing FFTs. There can be some pre-computed state to the FFT so its advantageous
    /// to save the fft setup if you are going to be doing multiple FFTs, hence why this is a struct and
    /// not a function. This implementation is only used for std environments; a different
    /// implementation is used for embedded cortex targets.
    pub struct Fft {
        fft: Arc<dyn RealToComplex<f32>>
    }

    impl Fft {
        pub fn new(size: usize) -> Self {
            let mut fft_planner = RealFftPlanner::<f32>::new();
            Self {
                fft: fft_planner.plan_fft_forward(size)
            }
        }

        /// Compute FFT of input data, storing to output
        /// Input data is modified in the process
        pub fn process(&mut self, data: &mut [f32], output: &mut [Complex<f32>]) {
            assert!(data.len() == self.fft.len());
            assert!(output.len() == self.fft.len() / 2 + 1);

            self.fft.process(data, output).unwrap();
            for i in 0..output.len() {
                output[i] /= self.fft.len() as f32;
            }
        }
    }

}

#[cfg(feature="std")]
pub mod fftimpl {
    use rustfft::FftPlanner;
    use std::sync::Arc;
    use super::Complex;

    pub struct Fft {
        fft: Arc<dyn rustfft::Fft<f32>>,
    }

    impl Fft {
        pub fn new(size: usize) -> Self {
            let mut planner = FftPlanner::<f32>::new();
            Self {
                fft: planner.plan_fft_forward(size),
            }
        }

        pub fn len(&self) -> usize {
            self.fft.len()
        }

        pub fn process(&mut self, data: &mut [f32], output: &mut [Complex<f32>]) {
            assert!(data.len() == self.fft.len());
            assert!(output.len() == self.fft.len() / 2 + 1);

            let mut buf: Vec<Complex<f32>> = data.iter().cloned().map(|x| Complex { re: x, im: 0.0f32 }).collect();
            self.fft.process(&mut buf);
            // Copy first N/2 + 1 components
            output.copy_from_slice(&buf[0..output.len()]);
            for i in 0..output.len() {
                output[i] /= self.fft.len() as f32;
            }
        }
    }
}

#[cfg(not(feature="std"))]
pub mod fftimpl {

}

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
                let mut complex_sum = Complex::<f32>::zero();
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
    pub fn new() -> Self {
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
                let mut complex_sum = Complex::<f32>::zero();
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


pub struct FftProcessor {
    fft: fftimpl::Fft,
}

impl FftProcessor {
    pub fn new(window_size: usize) -> Self {
        Self { fft: fftimpl::Fft::new(window_size) }
    }

    /// Compute spectra for a chunk of sample data
    /// The channel data is modified in the process, so this is destructive
    pub async fn compute_ffts(
        &mut self,
        input: &mut dyn SampleBuffer,
        output: &mut dyn Spectra,
    )
    {
        assert!(input.channels() == output.channels());
        assert!(input.len() == self.fft.len());
        assert!(output.nfft() == self.fft.len() / 2 + 1);

        let nfft = output.nfft();

        let mut spectra = output.as_array_mut().unwrap();

        spectra.fill(Complex { re: 0.0, im: 0.0 });

        for ch in 0..input.channels() {
            let in_samples = input.get_mut(ch).unwrap();
            let mut out_row = spectra.row_mut(ch);

            self.fft.process(in_samples, out_row.as_slice_mut().unwrap());
            for i in 0..nfft {
                out_row[i] /= self.fft.len() as f32;
            }
            // Yield to executor between channels to minimize the amount of time we block other processing tasks
            yield_now().await;
        }
    }
}



