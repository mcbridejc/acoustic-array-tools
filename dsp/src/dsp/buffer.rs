
use heapless::pool::singleton::{Box, Pool};
use core::mem::MaybeUninit;
use num_complex::Complex;
use ndarray::{ShapeBuilder, ArrayViewMut2, ArrayView2, Array2};

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

pub trait SampleBuffer {
    /// Returns true if this buffer has valid samples associated with it
    fn data_valid(&self) -> bool;

    /// Get data for a single channel as a mutable slice
    ///
    /// The slice will have `len()` elements
    ///
    /// Panics if ch is greater than (channels() - 1)
    fn get_mut<'a>(&'a mut self, ch: usize) -> Option<&'a mut [f32]>;

    /// Get data for a single channel as an immutable slice
    ///
    /// The slice will have 'len()' elements
    ///
    /// Panics if ch is greater than (channels() - 1)
    fn get<'a>(&'a mut self, ch: usize) -> Option<&'a [f32]>;

    /// Returns the number of channels held by this
    fn channels(&self) -> usize;

    /// Returns the number of samples stored for each channel
    fn len(&self) -> usize;
}

pub struct PoolSampleBuffer<C, const NCHAN: usize>
where
    C: Pool,
{
    pub pcm: [Option<Box<C>>; NCHAN],
    pub rms: f32,
    pub index: u32,
}

impl<C, const NCHAN: usize> PoolSampleBuffer<C, NCHAN>
where
    C: Pool,
{
    const INIT: Option<Box<C>> = None;
    pub fn new() -> Self {
        Self {
            pcm: [Self::INIT; NCHAN],
            rms: 0.0,
            index: 0
        }
    }
}

impl<C, const WINDOW_SIZE: usize, const NCHAN: usize> SampleBuffer for PoolSampleBuffer<C, NCHAN>
where
    C: Pool<Data = MaybeUninit<[f32; WINDOW_SIZE]>>,
{
    fn get_mut<'a>(&'a mut self, ch: usize) -> Option<&'a mut [f32]> {
        if ch > NCHAN {
            panic!("Accessed out of range channel {}", ch);
        }
        match self.pcm[ch].as_mut() {
            Some(sample_box) => {
                unsafe {
                    //let dataslice = sample_box.assume_init_mut();
                    //Some(ArrayViewMut1::from_shape(ShapeBuilder::into_shape((WINDOW_SIZE,)), dataslice))
                    Some(sample_box.assume_init_mut())
                }
            },
            None => None
        }
    }

    fn get<'a>(&'a mut self, ch: usize) -> Option<&'a [f32]> {
        if ch > NCHAN {
            panic!("Accessed out of range channel {}", ch);
        }
        match self.pcm[ch].as_mut() {
            Some(sample_box) => {
                unsafe {
                    //let dataslice = sample_box.assume_init_mut();
                    //Some(ArrayViewMut1::from_shape(ShapeBuilder::into_shape((WINDOW_SIZE,)), dataslice))
                    Some(sample_box.assume_init_ref())
                }
            },
            None => None
        }
    }

    fn channels(&self) -> usize {
        NCHAN
    }

    fn len(&self) -> usize {
        WINDOW_SIZE
    }

    fn data_valid(&self) -> bool {
        // Assuming here that if pcm[0] is not present, all channels are not present which should be
        // enforced throughout the code.
        self.pcm[0].is_some()
    }
}


pub trait Spectra {
    /// True if valid spectra data is contained
    fn data_valid(&self) -> bool;

    /// Get the number of channels in the set of spectra
    fn channels(&self) -> usize;

    // Get the size of each spectra
    fn nfft(&self) -> usize;

    /// Get spectra as a channels x nfft mutable array
    fn as_array_mut(&mut self) -> Option<ArrayViewMut2<Complex<f32>>>;

    /// Get spectra as a channels x nfft array
    fn as_array(&self) -> Option<ArrayView2<Complex<f32>>>;

    /// Get the RMS for the sample window used to generate this spectra
    fn rms(&self) -> f32;

    fn set_rms(&mut self, value: f32);

    /// Get the index for the sample window used to generate this spectra
    fn index(&self) -> u32;

    fn set_index(&mut self, value: u32);

    /// Get the average of the magnitude of all channels
    ///
    /// Result stored to out
    ///
    /// Panics if the length of out is not equal to nfft
    fn avg_mag(&self, out: &mut [f32]) {
        if out.len() != self.nfft() {
            panic!("NFFT is {}, but output buffer provided is {}.", self.nfft(), out.len());
        }

        for i in 0..out.len() {
            out[i] = 0.0;
        }

        if let Some(s) = self.as_array() {
            for i in 0..self.nfft()  {
                for ch in 0..self.channels() {
                    out[i] += s[[ch, i]].norm();
                }
                out[i] /= self.channels() as f32;
                out[i] = 20.0 * libm::log10f(out[i]);
            }
        }
    }
}

/// Implements a Spectra object using dynamic heap allocation
pub struct HeapSpectra {
    spectra: Array2<Complex<f32>>,
    rms: f32,
    index: u32,
}

impl HeapSpectra {
    pub fn new(nfft: usize, nchan: usize) -> Self {
        Self {
            spectra: Array2::zeros((nchan, nfft)),
            rms: 0.0,
            index: 0,
        }
    }
}

impl Spectra for HeapSpectra {
    fn rms(&self) -> f32 {
        self.rms
    }

    fn set_rms(&mut self, value: f32) {
        self.rms = value;
    }

    fn index(&self) -> u32 {
        self.index
    }

    fn set_index(&mut self, value: u32) {
        self.index = value;
    }

    fn data_valid(&self) -> bool {
        true
    }

    fn channels(&self) -> usize {
        self.spectra.dim().0
    }

    fn nfft(&self) -> usize {
        self.spectra.dim().1
    }

    fn as_array(&self) -> Option<ArrayView2<Complex<f32>>> {
        Some(self.spectra.view())
    }

    fn as_array_mut(&mut self) -> Option<ArrayViewMut2<Complex<f32>>> {
        Some(self.spectra.view_mut())
    }
}

/// Implements a Spectra object with a fixed size and no heap allocation
pub struct StaticSpectra<const NFFT: usize, const NCHAN: usize> {
    spectra: Option<[[Complex<f32>; NFFT]; NCHAN]>,
    rms: f32,
    index: u32,
}

impl<const NFFT: usize, const NCHAN: usize> StaticSpectra<NFFT, NCHAN> {

    pub fn blank() -> Self {
        Self {
            spectra: None, //[[Complex { re: 0.0, im: 0.0 }; NFFT]; NCHAN],
            rms: 0.0,
            index: 0,
        }
    }
}

impl<const NFFT: usize, const NCHAN: usize> Spectra for StaticSpectra<NFFT, NCHAN> {
    fn data_valid(&self) -> bool {
        self.spectra.is_some()
    }

    fn channels(&self) -> usize {
        NCHAN
    }

    fn nfft(&self) -> usize {
        NFFT
    }

    fn index(&self) -> u32 {
        self.index
    }

    fn set_index(&mut self, value: u32) {
        self.index = value;
    }

    fn rms(&self) -> f32 {
        self.rms
    }

    fn set_rms(&mut self, value: f32) {
        self.rms = value;
    }

    fn as_array(&self) -> Option<ArrayView2<Complex<f32>>> {
        match self.spectra {
            Some(s) => unsafe {
                Some(ArrayView2::from_shape_ptr(ShapeBuilder::into_shape((NCHAN, NFFT)), core::ptr::addr_of!(s) as *const Complex<f32>))
            },
            None => None
        }
    }

    fn as_array_mut(&mut self) -> Option<ArrayViewMut2<Complex<f32>>> {
        match self.spectra {
            Some(s) => unsafe {
                Some(ArrayViewMut2::from_shape_ptr(ShapeBuilder::into_shape((NCHAN, NFFT)), core::ptr::addr_of!(s) as *mut Complex<f32>))
            },
            None => None
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