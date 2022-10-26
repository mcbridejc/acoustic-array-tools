
use std::{boxed::Box, ops::DerefMut};
use core::cell::RefCell;
use heapless::spsc::{Queue, Producer, Consumer};

use dsp::{
    beamforming::{BeamFormer, StaticBeamFormer},
    buffer::{PdmBuffer, Spectra},
    fft::{
        FftProcessor,
        fftimpl::Fft,
    },
    pdm_processing::{StaticPdmProcessor},
    pipeline::{preprocess, process_spectra, ProcessingQueue},
};

use num_complex::Complex;

pub const FSAMPLE: f32 = 24e3;
const WINDOW_SIZE: usize = 1024;
pub const NFFT: usize = WINDOW_SIZE / 2 + 1;
pub const NUM_CHANNELS: usize = 6;
const PDM_QUEUE_SIZE: usize = 16;
const PROCESS_QUEUE_SIZE: usize = 128;

const DECIMATION: usize = 128;
const PDM_BUFFER_SIZE: usize = NUM_CHANNELS * WINDOW_SIZE * DECIMATION / 8;
const NUM_PDM_BUFFERS: usize = 7;
const SAMPLE_STORAGE_SIZE:usize = 4 * WINDOW_SIZE * (NUM_PDM_BUFFERS + NUM_CHANNELS);

use core::mem::MaybeUninit;
use heapless::{pool, pool::singleton::Pool};

pool!(PDMPOOL: MaybeUninit<[u8; PDM_BUFFER_SIZE]>);
pool!(PCMPOOL: MaybeUninit<[f32; WINDOW_SIZE]>);

static mut PROCESSOR_INIT: bool = false;

pub fn processors() -> (UdpToPdmBuffer, Processor) {

    static mut PDM_STORAGE: [u8; PDM_BUFFER_SIZE * NUM_PDM_BUFFERS] = [0; PDM_BUFFER_SIZE * NUM_PDM_BUFFERS];
    static mut PCM_STORAGE: [u8; SAMPLE_STORAGE_SIZE] = [0; SAMPLE_STORAGE_SIZE];

    let inited = unsafe { PROCESSOR_INIT };
    if inited {
        panic!("Processor uses static storage; you can't create multiple");
    }
    unsafe {
        PDMPOOL::grow(&mut PDM_STORAGE);
        PCMPOOL::grow(&mut PCM_STORAGE);
    }

    static mut QUEUE: Queue<PdmBuffer<PDMPOOL>, PDM_QUEUE_SIZE> = Queue::new();
    let (tx, rx) = unsafe { QUEUE.split() };
    let pdm = UdpToPdmBuffer::new(tx);
    let processor = Processor::new(rx);

    (pdm, processor)
}

pub struct UdpToPdmBuffer
{
    working_buffer: Vec<u8>,
    ready_buffer_tx: Producer<'static, PdmBuffer<PDMPOOL>, PDM_QUEUE_SIZE>,
}

impl UdpToPdmBuffer
{
    pub fn new(tx: Producer<'static, PdmBuffer<PDMPOOL>, PDM_QUEUE_SIZE>) -> Self {
        Self {
            working_buffer: Vec::new(),
            ready_buffer_tx: tx,
        }
    }

    pub fn len(&self) -> usize {
        self.ready_buffer_tx.len()
    }

    /// Collect arbitrarily sized chunks of pdm bytes into window sized buffers and pass them off to
    /// a channel for processing
    pub fn push_pdm_chunk(&mut self, packet: &[u8])
    {
        let bytes_remaining = PDM_BUFFER_SIZE - self.working_buffer.len();
        let mut offset = 0;
        if packet.len() >= bytes_remaining {
            // Enough bytes to complete this buffer. Finish filling it, and process it
            offset = bytes_remaining;
            self.working_buffer.extend_from_slice(&packet[0..bytes_remaining]);

            if let Some(pdm_data_box) = PDMPOOL::alloc() {
                let mut pdm_data_box = pdm_data_box.init(MaybeUninit::uninit());
                let pdm_data = unsafe { pdm_data_box.assume_init_mut() };
                pdm_data.copy_from_slice(&self.working_buffer);
                let pdm_buffer = PdmBuffer::new(0, pdm_data_box);
                // Queue for further processing
                match self.ready_buffer_tx.enqueue(pdm_buffer) {
                    Ok(..) => (),
                    Err(..) => println!("Overran PDM buffer"),
                }
            } else {
                println!("Exhausted PDMPOOL. Dropping buffer.")
            }

            // Create new buffer to continue filling
            self.working_buffer = Vec::with_capacity(PDM_BUFFER_SIZE);
        }

        // Store any bytes remaining after completing a buffer
        if offset < packet.len() {
            self.working_buffer.extend_from_slice(&packet[offset..]);
        }
    }
}

pub const N_AZ_POINTS: usize = 100;
pub const IMAGE_GRID_RES: usize = 20;

pub struct Processor {
    pub rms_series: Vec<f32>,
    pub latest_avg_spectrum: [f32; NFFT],
    buffer_rx: RefCell<Consumer<'static, PdmBuffer<PDMPOOL>, PDM_QUEUE_SIZE>>,
    processing_queue: ProcessingQueue<PDMPOOL, PCMPOOL, PROCESS_QUEUE_SIZE>,
    pdm_processor: StaticPdmProcessor<NUM_CHANNELS>,
    fft_processor: RefCell<Fft>,
    az_beamformer: Box<StaticBeamFormer<NUM_CHANNELS, N_AZ_POINTS, NFFT>>,
    image_beamformer: Box<StaticBeamFormer<NUM_CHANNELS, {IMAGE_GRID_RES*IMAGE_GRID_RES}, NFFT>>,
}

pub unsafe fn unsafe_allocate<T>() -> Box<T> {
    let grid_box: Box<T>;

    use std::alloc::{alloc, Layout};
    let layout = Layout::new::<T>();
    let ptr = alloc(layout) as *mut T;
    grid_box = Box::from_raw(ptr);

    return grid_box;
}

impl Processor {
    pub fn new(buffer_rx: Consumer<'static, PdmBuffer<PDMPOOL>, PDM_QUEUE_SIZE>) -> Self {
        // Define a set of focal points arranged in a circle around the origin for determining
        // azimuth to the source
        let az_focal_points = make_circular_focal_points::<N_AZ_POINTS>(1.0, 0.1);
        // Define a grid of focal points above the origin for forming an image fo the sources
        let image_focal_points = make_grid_focal_points::<IMAGE_GRID_RES, {IMAGE_GRID_RES*IMAGE_GRID_RES}>(1.0, 0.25);
        // Define microphone locations relative to the origin, in meters
        let mics = [
            [-0.05088, 0.0294, 0.],
            [-0.05088, -0.0294, 0.],
            [0.05088, 0.0294, 0.],
            [0.0, 0.05875, 0.],
            [0.05088, -0.0294, 0.],
            [0., -0.05875, 0.],
        ];

        // I had to do some shenanigans to prevent this large allocation from blowing up the stack.
        //
        // In some circumstances, it will first allocate on the stack and later memcpy to the heap
        // allocation. If BeamFormer::new took arguments and constructed the internal
        // steering_vector matrix, it would do a stack allocation in both release and debug builds.
        // In the commented case below, where new() does no initialization and setup is later
        // called, it overflows in debug build only. To be safe in both, I had to do the
        // unsafe_allocate approach. The box_syntax feature appears to address this, but it also
        // appears to have been dropped will never go stable.
        let mut az_beamformer = unsafe { unsafe_allocate::<StaticBeamFormer<NUM_CHANNELS, N_AZ_POINTS, NFFT>>() };
        let mut image_beamformer = unsafe { unsafe_allocate::<StaticBeamFormer<NUM_CHANNELS, {IMAGE_GRID_RES * IMAGE_GRID_RES}, NFFT>>() };
        // This causes stack overflow in debug builds
        //let mut az_beamformer = Box::new(BeamFormer::new());
        //let mut image_beamformer = Box::new(BeamFormer::new());

        az_beamformer.setup(mics, az_focal_points, FSAMPLE);
        image_beamformer.setup(mics, image_focal_points, FSAMPLE);

        Self {
            rms_series: vec![0.0; 200],
            latest_avg_spectrum: [0.0; NFFT],
            buffer_rx: RefCell::new(buffer_rx),
            pdm_processor: StaticPdmProcessor::new(),
            fft_processor: RefCell::new(Fft::new(WINDOW_SIZE)),
            processing_queue: ProcessingQueue::new(),
            az_beamformer,
            image_beamformer,
        }
    }

    pub async fn stage1(&self, rms_threshold: f32) {
        preprocess(
            &self.pdm_processor,
            &mut self.buffer_rx.borrow_mut(),
            &self.processing_queue,
            rms_threshold
        ).await;

    }

    pub async fn stage2(
        &self,
        spectra_out: &mut dyn Spectra,
    ) -> bool
    {
        // TODO: Fix the generic inferrence. I broke the inferrence of NCHAN when I re-factored for
        // heap based buffers, and apparently if rustc can't infer one const generic it can't infer
        // any?
        process_spectra::<_, _, PROCESS_QUEUE_SIZE, PDM_BUFFER_SIZE, WINDOW_SIZE, NUM_CHANNELS, NFFT>(
            &self.pdm_processor,
            self.fft_processor.borrow_mut().deref_mut(),
            &self.processing_queue,
            spectra_out
        ).await
    }

    pub fn beamform_power(
        &self,
        spectra: & dyn Spectra,
        start_freq: f32,
        end_freq: f32,
    ) -> ([f32; N_AZ_POINTS], [f32; IMAGE_GRID_RES * IMAGE_GRID_RES] )
    {
        let mut az_powers = [0.0; N_AZ_POINTS];
        let mut image_powers = [0.0; IMAGE_GRID_RES * IMAGE_GRID_RES];
        self.az_beamformer.compute_power(spectra, &mut az_powers, start_freq, end_freq);
        self.image_beamformer.compute_power(spectra, &mut image_powers, start_freq, end_freq);
        (az_powers, image_powers)
    }
}

pub fn weighted_azimuth(az_powers: &[f32]) -> Complex<f32> {
    let mut result = Complex{re: 0.0, im: 0.0};
    for i in 0..az_powers.len() {
        let theta = std::f32::consts::PI * 2.0 * i as f32 / az_powers.len() as f32;
        result += Complex::from_polar(az_powers[i], theta);
    }
    result
}


#[derive(Default, Clone, Copy)]
pub struct AzDatapoint {
    moment: Complex<f32>,
    rms: f32,
}
pub struct AzFilter<const DEPTH: usize> {
    data: [AzDatapoint; DEPTH],
    inptr: usize,
}

impl<const DEPTH: usize> AzFilter<DEPTH> {
    pub fn new() -> Self {
        Self {
            data: [AzDatapoint{ moment: Complex {re: 0.0, im: 0.0}, rms: 0.0 }; DEPTH],
            inptr: 0,
        }
    }

    pub fn push(&mut self, moment: Complex<f32>, rms: f32) -> Option<f32> {
        // If the new sample is the loudest in the queue, use it's direction???
        let mut max_rms: f32 = -60.0;
        for d in &self.data {
            if d.rms > max_rms {
                max_rms = d.rms;
            }
        }

        self.data[self.inptr] = AzDatapoint{ moment, rms };
        self.inptr = (self.inptr + 1) % DEPTH;
        if rms > max_rms && moment.norm() > 30. {
            // Return the angle of the current moment
            return Some(moment.to_polar().1);
        }

        None
    }
}

pub struct AzFilter2 {
    sample_count: usize,
    moment: Complex<f32>,
    dynamic_threshold: f32,
    rms_accum: f32,
}

impl AzFilter2 {
    const RMS_THRESH: f32 = -55.;
    const RMS_DECAY: f32 = 0.25;

    pub fn new() -> Self {
        Self {
            sample_count: 0,
            moment: Complex { re: 0.0, im: 0.0 },
            dynamic_threshold: Self::RMS_THRESH,
            rms_accum: 0.0,
        }
    }

    pub fn push(&mut self, moment: Complex<f32>, rms: f32) -> Option<f32> {
        if self.dynamic_threshold > Self::RMS_THRESH {
            self.dynamic_threshold -= Self::RMS_DECAY;
        }
        if self.dynamic_threshold < Self::RMS_THRESH {
            self.dynamic_threshold = Self::RMS_THRESH;
        }

        if rms >= self.dynamic_threshold {
            self.sample_count += 1;
            self.moment += moment;
            self.rms_accum += rms;
            if self.sample_count > 25 {
                return self.complete();
            }
        } else {
            if self.sample_count > 0 {
                return self.complete();
            }
        }

        None
    }

    pub fn complete(&mut self) -> Option<f32> {
        let (mag, angle) = self.moment.to_polar();
        let result = if mag > 30.0 {
            Some(angle)
        } else {
            None
        };

        self.dynamic_threshold = self.rms_accum / self.sample_count as f32;
        self.sample_count = 0;
        self.moment = Complex{ re: 0.0, im: 0.0 };
        self.rms_accum = 0.0;

        result
    }
}


pub fn make_circular_focal_points<const N: usize>(radius: f32, z: f32) -> [[f32; 3]; N] {
    let mut points = [[0.0; 3]; N];

    for i in 0..N {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / N as f32;
        points[i][0] = f32::sin(theta) * radius;
        points[i][1] = f32::cos(theta) * radius;
        points[i][2] = z;
    }
    points
}

pub fn make_grid_focal_points<const N: usize, const M: usize>(width: f32, z: f32) -> [[f32; 3]; M]
{
    // This feels terrible, but I cannot find a way to size the array without passing two generic arguments
    assert!(N*N == M);
    let mut points = [[0.0; 3]; M];

    let mut i = 0;
    for iy in 0..N {
        let y = -width / 2.0 + width * iy as f32 / (N - 1) as f32;
        for ix in 0..N {
            let x = -width / 2.0 + width * ix as f32 / (N - 1) as f32;
            points[i][0] = x;
            points[i][1] = y;
            points[i][2] = z;
            i += 1;
        }
    }
    points
}

