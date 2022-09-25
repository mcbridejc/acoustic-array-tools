use core::mem::MaybeUninit;

use crate::buffer::SampleBuffer;
use num_complex::Complex;
use heapless::pool::singleton::Pool;
use ndarray::{ArrayBase, ArrayView3, Axis};
use realfft::num_traits::Zero;


const SPEED_OF_SOUND: f32 = 343.0;


#[cfg(feature="std")]
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

#[cfg(not(feature="std"))]
pub mod fftimpl {

}

pub struct BeamFormer<const NCHAN: usize, const NFOCAL: usize, const NFFT: usize> {
    // When generic const exprs goes stable, this could just be an array of length {NCHAN * NFOCAL *
    // NFFT}
    steering_vectors: [[[Complex<f32>; NFFT]; NFOCAL]; NCHAN],
    sample_freq: f32,
}

impl<const NCHAN: usize, const NFOCAL: usize, const NFFT: usize> BeamFormer<NCHAN, NFOCAL, NFFT> {
    pub fn new() -> Self {
        let bf = BeamFormer{
            steering_vectors: [[[Complex::<f32>{re: 0.0, im: 0.0}; NFFT]; NFOCAL]; NCHAN],
            sample_freq: 0.0,
        };

        // for i in 0..NCHAN {
        //     for j in 0..NFOCAL {
        //         let m = mics[i];
        //         let fp = focal_points[j];
        //         let mut sum_sqr: f32 = 0.0;
        //         for dim in 0..3 {
        //             let x = m[dim] - fp[dim];
        //             sum_sqr += x * x;
        //         }
        //         let d = libm::sqrtf(sum_sqr);
        //         for k in 0..NFFT {
        //             // Center frequency of the FFT bin
        //             let bin_freq = k as f32 * sample_freq / NFFT as f32;
        //             // phase shift at center frequency based on distance between source and mic
        //             let angle = core::f32::consts::PI * 2.0f32 * bin_freq * d / SPEED_OF_SOUND;
        //             bf.steering_vectors[i][j][k] = Complex::from_polar(1.0, angle);
        //         }
        //     }
        // }

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

    pub fn sv_array(&self) -> ArrayView3<Complex<f32>> {
        use ndarray::ShapeBuilder;
        unsafe { 
            ArrayView3::from_shape_ptr(
                (NCHAN, NFOCAL, NFFT).into_shape(),
                self.steering_vectors.as_ptr() as *const Complex<f32>
            )
        }
    }

    pub fn compute_power(&self, spectra: &Spectra<NFFT, NCHAN>, start_freq: f32, end_freq: f32) -> [f32; NFOCAL] {

        let freq_bin_start = libm::floorf(2.0 * start_freq * NFFT as f32 / self.sample_freq) as usize;
        let freq_bin_end = f32::ceil(2.0 * end_freq * NFFT as f32 / self.sample_freq) as usize;

        assert!(freq_bin_end >= freq_bin_start);

        let mut power = [0.0f32; NFOCAL];
        
        for i in 0..NFOCAL {
            let mut power_sum: f32 = 0.0;
            for bin in freq_bin_start..(freq_bin_end + 1) {
                let mut complex_sum = Complex::<f32>::zero();
                for ch in 0..NCHAN {
                    complex_sum += self.steering_vectors[ch][i][bin] * spectra.spectra[ch][bin];
                }
                power_sum += complex_sum.norm();
                
            }
            power[i] = power_sum / (freq_bin_start - freq_bin_end + 1) as f32;
            power[i] = 20.0 * libm::log10f(power[i]);
        }
        power
    }
}

pub struct Spectra<const NFFT: usize, const NCHAN: usize> {
    spectra: [[Complex<f32>; NFFT]; NCHAN],
}


impl<const NFFT: usize, const NCHAN: usize> Spectra<NFFT, NCHAN> {

    pub fn blank() -> Self {
        Self { spectra: [[Complex::zero(); NFFT]; NCHAN] }
    }

    pub fn avg_mag(&self) -> [f32; NFFT] {
        let mut avg = [0.0; NFFT];
        
        for i in 0..NFFT  {
            for ch in 0..NCHAN {
                avg[i] += self.spectra[ch][i].norm();
            }
            avg[i] /= NCHAN as f32;
            avg[i] = 20.0 * libm::log10f(avg[i]);
        }
        avg
    }
}

pub struct FftProcessor<const WINDOW_SIZE: usize, const NFFT: usize> {
    fft: fftimpl::Fft,
}

impl<const WINDOW_SIZE: usize, const NFFT: usize> FftProcessor<WINDOW_SIZE, NFFT> {
    pub fn new() -> Self {
        Self { fft: fftimpl::Fft::new(WINDOW_SIZE) }
    }

    /// Compute spectra for a chunk of sample data
    /// The channel data is modified in the process, so this is destructive
    pub fn compute_ffts<C, const NCHAN: usize>(
        &mut self,
        input: &mut SampleBuffer<C, NCHAN>,
        output: &mut Spectra<NFFT, NCHAN>
    ) 
    where 
        C: Pool<Data = MaybeUninit<[f32; WINDOW_SIZE]>>,
    {
        for ch in 0..NCHAN {
            let in_sample_box = input.pcm[ch].as_mut().unwrap();
            let in_samples = unsafe { in_sample_box.assume_init_mut() };
            self.fft.process(in_samples, &mut output.spectra[ch]);
            for i in 0..output.spectra.len() { 
                output.spectra[ch][i] /= WINDOW_SIZE as f32;
            }
        }
    }
}

