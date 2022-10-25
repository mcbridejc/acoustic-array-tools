#![cfg_attr(not(feature="std"), no_std)]
#![allow(dead_code)]
#![feature(generic_const_exprs)]

pub mod azimuth;
pub mod beamforming;
pub mod buffer;
pub mod cic;
pub mod fft;
pub mod fir;
#[cfg(feature="std")]
pub mod generation;
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
    pub const fn new() -> Self {
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

