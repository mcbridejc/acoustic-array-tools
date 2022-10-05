#![cfg_attr(not(feature="std"), no_std)]


#[cfg(feature="std")]
pub use cortex_m::interrupt::Mutex;

#[cfg(not(feature="std"))]
pub use std::sync::Mutex;

pub mod beamforming;
pub mod buffer;
pub mod cic;
pub mod dasp;
pub mod fir;
pub mod pdm_processing;
pub mod pipeline;

// Simple moving average decimator
#[derive(Clone, Copy)]
pub struct Decimator<const DECIMATION: usize> {
    accum: f32,
    pos: usize,
}

impl<const DECIMATION: usize> Decimator<DECIMATION>
{
    pub fn new() -> Self {
        Self { accum: 0., pos: 0 }
    }

    pub fn process_sample<F>(&mut self, sample: f32, mut output: F)
    where F: FnMut(f32)
    {
        self.accum += sample;
        self.pos += 1;
        if self.pos == DECIMATION {
            self.pos = 0;
            output(self.accum / DECIMATION as f32);
            self.accum = 0.;
        }
    }
}

