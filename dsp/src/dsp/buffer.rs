
use heapless::pool::singleton::{Box, Pool};
use core::mem::MaybeUninit;
use num_complex::Complex;

pub struct PdmBuffer<D>
where D: heapless::pool::singleton::Pool
{
    pub pdm_data: Box<D>,
    pub index: u32,
}

impl<D, const N: usize> PdmBuffer<D>
where D: Pool<Data = MaybeUninit<[u8; N]>>
{
    pub fn new(index: u32, pdm_data: Box<D>) -> Self {pub struct Spectra<const NFFT: usize, const NCHAN: usize> {
        pub spectra: [[Complex<f32>; NFFT]; NCHAN],
        pub rms: f32,
        pub index: u32,
    }
        Self {
            pdm_data,
            index
        }
    }
}

pub struct RmsBuffer<D, C>
where
    D: Pool,
    C: Pool,
{
    pub pdm_data: Option<Box<D>>,
    pub ch1_pcm: Option<Box<C>>,
    pub rms: f32,
    pub index: u32,
}

impl<D, C> RmsBuffer<D, C>
where
    D: Pool,
    C: Pool,
{
    pub fn new() -> Self {
        Self {
            pdm_data: None,
            ch1_pcm: None,
            rms: 0.0,
            index: 0
        }
    }
}

pub struct SampleBuffer<C, const NUM_CHANNELS: usize>
where
    C: Pool,
{
    pub pcm: [Option<Box<C>>; NUM_CHANNELS],
    pub rms: f32,
    pub index: u32,
}

impl<C, const NUM_CHANNELS: usize> SampleBuffer<C, NUM_CHANNELS>
where
    C: Pool,
{
    const INIT: Option<Box<C>> = None;
    pub fn new() -> Self {
        Self {
            pcm: [Self::INIT; NUM_CHANNELS],
            rms: 0.0,
            index: 0
        }
    }
}

pub struct Spectra<const NFFT: usize, const NCHAN: usize> {
    pub spectra: Option<[[Complex<f32>; NFFT]; NCHAN]>,
    pub rms: f32,
    pub index: u32,
}

impl<const NFFT: usize, const NCHAN: usize> Spectra<NFFT, NCHAN> {

    pub fn blank() -> Self {
        Self {
            spectra: None, //[[Complex { re: 0.0, im: 0.0 }; NFFT]; NCHAN],
            rms: 0.0,
            index: 0,
        }
    }

    /// Return a spectrum with averate of the magnitude of all spectra, if they are available.
    /// Returns None if spectra is None.
    pub fn avg_mag(&self) -> Option<[f32; NFFT]> {
        let mut avg = [0.0; NFFT];

        match self.spectra {
            Some(s) => {
                for i in 0..NFFT  {
                    for ch in 0..NCHAN {
                        avg[i] += s[ch][i].norm();
                    }
                    avg[i] /= NCHAN as f32;
                    avg[i] = 20.0 * libm::log10f(avg[i]);
                }
                Some(avg)
            },
            None => {
                None
            }
        }
    }
}

/// Represent the result of beamforming power calculation at a set of NFOCAL focal points. Because
/// of CPU constraints, power is optional and may not be present if the input data was dropped
/// before it could be processed.
pub struct PowerResult<const NFOCAL: usize> {
    power: Option<[f32; NFOCAL]>,
    rms: f32,
    index: u32,
}

impl<const NFOCAL: usize> PowerResult<NFOCAL> {
    pub fn new() -> Self {
        Self {
            power: None,
            rms: 0.0,
            index: 0,
        }
    }
}