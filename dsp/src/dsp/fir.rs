

// TODO: Storing coeffs by value here is wasting memory when multiple channels are processed with
// the same filter We can store one reference; it's probably static, and even if not we should be
// able to ensure the coeffs outlive the filter

#[derive(Clone, Copy)]
pub struct FloatFir<const TAPS: usize> {
    coeffs: [f32; TAPS],
    samples: [f32; TAPS],
    pos: usize
}

impl<const TAPS: usize> FloatFir<TAPS> {
    pub const fn new(coeffs: [f32; TAPS]) -> Self {
        Self {
            coeffs: coeffs,
            samples: [0.0; TAPS],
            pos: 0
        }
    }

    pub fn process_sample(&mut self, sample: f32) -> f32 {
        self.samples[self.pos] = sample;
        self.pos = (self.pos + 1) % TAPS;
        self.output()
    }

    pub fn output(&self) -> f32 {
        let mut y = 0.0f32;

        for i in 0..TAPS {
            let sample_idx = (self.pos + i) % TAPS;
            y += self.coeffs[i] * self.samples[sample_idx];
        }
        y
    }
}

/// FIR Filter with decimation
pub use firimpl::FloatFirDecimate;

// TODO: The cmsis-dsp implementation below does not work correctly. Needs work.
// The noise floor goes way up using it, and it's going to take some setup work
// to debug. The rust implementation below works fine, just probably slower.

// #[cfg(feature="cortex-m7")]
// mod firimpl {
//     use cmsis_dsp::filtering::FloatFirDecimate as CmsisFirDecimate;
//     pub struct FloatFirDecimate<
//         const TAPS: usize,
//         const BLKSIZE: usize,
//         const DEC: usize
//     >
//     where
//         [(); TAPS + BLKSIZE - 1]:
//     {
//         filter_state: [f32; TAPS + BLKSIZE - 1],
//         coeffs: [f32; TAPS],
//     }

//     impl<
//         const TAPS: usize,
//         const BLKSIZE: usize,
//         const DEC: usize
//     > FloatFirDecimate<TAPS, BLKSIZE, DEC>
//     where
//         [(); TAPS + BLKSIZE - 1]:
//     {
//         pub fn new(coeffs: [f32; TAPS]) -> Self {
//             let mut revcoeffs = [0.0; TAPS];
//             for i in 0..TAPS {
//                 revcoeffs[i] = coeffs[TAPS - 1 - i];
//             }
//             Self {
//                 filter_state: [0.0; TAPS + BLKSIZE - 1],
//                 coeffs: revcoeffs,
//             }
//         }

//         pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
//             assert!(input.len() % DEC == 0);
//             assert!(output.len() >= input.len() / DEC);

//             let mut cmsis_state = CmsisFirDecimate::<'_, TAPS, DEC, BLKSIZE>::new(
//                 &self.coeffs,
//                 &mut self.filter_state
//             );
//             cmsis_state.run(input, output);
//         }
//     }
// }

// One day, if I figure out the CMSIS-DSP FIR decimate issues, this is std only
//#[cfg(feature="std")]
mod firimpl {
    #[derive(Clone, Copy)]
    pub struct FloatFirDecimate<
        const TAPS: usize,
        const BLKSIZE: usize,
        const DEC: usize
    > {
        samples: [f32; TAPS],
        coeffs: [f32; TAPS],
        pos: usize,
    }

    impl<
        const TAPS: usize,
        const BLKSIZE: usize,
        const DEC: usize
    > FloatFirDecimate<TAPS, BLKSIZE, DEC>
    {
        pub const fn new(coeffs: [f32; TAPS]) -> Self {
            Self {
                samples: [0.0; TAPS],
                coeffs,
                pos: 0,
            }
        }

        pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
            assert!(input.len() % DEC == 0);
            assert!(output.len() >= input.len() / DEC);


            let mut i = 0;
            while i < input.len() {
                // For performance sake, be sure we get rid of any bounds checking
                unsafe {
                    *self.samples.get_unchecked_mut(self.pos) = *input.get_unchecked(i);
                    self.pos = (self.pos + 1) % TAPS;

                    if i % DEC == DEC - 1 {
                        *output.get_unchecked_mut(i / DEC) = self.output();
                    }
                }
                i += 1;
            }
        }

        fn output(&self) -> f32 {
            let mut y = 0.0f32;

            for i in 0..TAPS {
                let sample_idx = (self.pos + i) % TAPS;
                y += self.coeffs[i] * self.samples[sample_idx];
            }
            y
        }
    }

}
