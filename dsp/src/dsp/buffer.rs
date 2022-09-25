
use heapless::pool::singleton::{Box, Pool};
use core::mem::MaybeUninit;

pub struct PdmBuffer<D>
where D: heapless::pool::singleton::Pool
{
    pub pdm_data: Box<D>,
    pub index: u32,
}

impl<D, const N: usize> PdmBuffer<D>
where D: Pool<Data = MaybeUninit<[u8; N]>>
{
    pub fn new(index: u32, pdm_data: Box<D>) -> Self {
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
