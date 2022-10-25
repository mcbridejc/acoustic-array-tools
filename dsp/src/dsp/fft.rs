use embassy_futures::yield_now;
use num_complex::Complex;

use crate::buffer::SampleBuffer;
use crate::buffer::Spectra;

pub trait FftProcessor {
    fn process(&mut self, data: &mut [f32], output: &mut [Complex<f32>]);
}

#[cfg(feature="std")]
pub mod fftimpl {
    use realfft::{RealFftPlanner, RealToComplex};
    use std::sync::Arc;
    use super::{Complex, FftProcessor};

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
    }

    impl FftProcessor for Fft {
        /// Compute FFT of input data, storing to output
        /// Input data is modified in the process
        fn process(&mut self, data: &mut [f32], output: &mut [Complex<f32>]) {
            assert!(data.len() == self.fft.len());
            assert!(output.len() == self.fft.len() / 2 + 1);

            self.fft.process(data, output).unwrap();
            for i in 0..output.len() {
                output[i] /= self.fft.len() as f32;
            }
        }
    }

}

// #[cfg(feature="std")]
// pub mod fftimpl {
//     use rustfft::FftPlanner;
//     use std::sync::Arc;
//     use num_complex::Complex;

//     use super::FftImplementation;

//     pub struct Fft {
//         fft: Arc<dyn rustfft::Fft<f32>>,
//     }

//     impl Fft {
//         pub fn new(size: usize) -> Self {
//             let mut planner = FftPlanner::<f32>::new();
//             Self {
//                 fft: planner.plan_fft_forward(size),
//             }
//         }

//         pub fn len(&self) -> usize {
//             self.fft.len()
//         }
//     }

//     impl FftImplementation for Fft {
//         fn process(&mut self, data: &mut [f32], output: &mut [Complex<f32>]) {
//             assert!(data.len() == self.fft.len());
//             assert!(output.len() == self.fft.len() / 2 + 1);

//             let mut buf: Vec<Complex<f32>> = data.iter().cloned().map(|x| Complex { re: x, im: 0.0f32 }).collect();
//             self.fft.process(&mut buf);
//             // Copy first N/2 + 1 components
//             output.copy_from_slice(&buf[0..output.len()]);
//             for i in 0..output.len() {
//                 output[i] /= self.fft.len() as f32;
//             }
//         }
//     }
// }

#[cfg(feature="cortex-m7")]
pub mod fftimpl {
    use core::cell::{RefCell, RefMut};

    use super::FftProcessor;
    use num_complex::Complex;
    use cmsis_dsp::transform::{Direction, FloatFft, OutputOrder};

    pub struct Fft<const WINDOW_SIZE: usize> {
        cmsisfft: RefCell<Option<FloatFft>>,
        buffer: [Complex<f32>; WINDOW_SIZE],
    }

    impl<const WINDOW_SIZE: usize> Fft<WINDOW_SIZE> {
        pub const fn new() -> Self {
            Self { 
                cmsisfft: RefCell::new(None),
                buffer: [Complex { re: 0.0, im: 0.0 }; WINDOW_SIZE],
            }
        }

        pub fn len(&self) -> usize {
            WINDOW_SIZE
        }

    }

    impl<const WINDOW_SIZE: usize> FftProcessor for Fft<WINDOW_SIZE> {
        fn process(&mut self, data: &mut[f32], output: &mut[Complex<f32>]) {
            // TODO: Doing an FFT this way with real data only is innefficient and we should write a
            // no_std implementation of the realfft crate here
            assert!(data.len() == WINDOW_SIZE);
            assert!(output.len() == WINDOW_SIZE / 2 + 1);
            for i in 0..WINDOW_SIZE {
                self.buffer[i] = Complex { re: data[i], im: 0.0 }
            }
            // Lazily instantiate if needed
            if self.cmsisfft.borrow().is_none() {
                self.cmsisfft.replace(Some(FloatFft::new(WINDOW_SIZE as u16).unwrap()));
            }
            let mut fftcell = self.cmsisfft.borrow_mut();
            let fft = fftcell.as_mut().unwrap();
            fft.run(&mut self.buffer[..], Direction::Forward, OutputOrder::Standard);
        }
    }
}

// trait FftProcessor {
//     fn compute_fft(
//         &mut self,
//         input: &mut dyn SampleBuffer,
//         output: &mut dyn Spectra,
//         channel: usize,
//     );
// }
// pub struct AdjustableFftProcessor {
//     fft: fftimpl::Fft,
// }

// impl AdjustableFftProcessor {
//     pub fn new(window_size: usize) -> Self {
//         Self { fft: fftimpl::Fft::new(window_size) }
//     }

//     /// Compute spectra for a chunk of sample data
//     /// The channel data is modified in the process, so this is destructive
//     pub async fn compute_ffts(
//         &mut self,
//         input: &mut dyn SampleBuffer,
//         output: &mut dyn Spectra,
//         channel: usize
//     )
//     {
//         assert!(input.channels() == output.channels());
//         assert!(input.len() == self.fft.len());
//         assert!(output.nfft() == self.fft.len() / 2 + 1);

//         let nfft = output.nfft();

//         let mut spectra = output.as_array_mut().unwrap();

//         spectra.fill(Complex { re: 0.0, im: 0.0 });

//         let in_samples = input.get_mut(channel).unwrap();
//         let mut out_row = spectra.row_mut(channel);

//         self.fft.process(in_samples, out_row.as_slice_mut().unwrap());
//         for i in 0..nfft {
//             out_row[i] /= self.fft.len() as f32;
//         }
//     }
// }

// /// FFT processor with window size known at compile time
// pub struct StaticFftProcessor<const NFFT: usize> {

// }