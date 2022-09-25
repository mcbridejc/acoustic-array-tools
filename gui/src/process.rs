
use std::rc::Rc;
use std::boxed::Box;
use crossbeam_channel::{unbounded, Sender, Receiver};
use hound;

use dsp::{
    buffer::{PdmBuffer, RmsBuffer, SampleBuffer},
    pdm_processing::{compute_rms_buffer, PdmProcessing, compute_sample_buffer},
    beamforming::{BeamFormer, Spectra, FftProcessor},
};

use realfft::{
    RealFftPlanner,
    num_complex::Complex,
};

const FSAMPLE: f32 = 24e3;
const WINDOW_SIZE: usize = 1024;
const NFFT: usize = WINDOW_SIZE / 2 + 1;
pub const NUM_CHANNELS: usize = 6;

const DECIMATION: usize = 128;
const PDM_BUFFER_SIZE: usize = NUM_CHANNELS * WINDOW_SIZE * DECIMATION / 8;
const SAMPLE_STORAGE_SIZE:usize = WINDOW_SIZE * (10 + NUM_CHANNELS * 8);

use core::mem::MaybeUninit;
use heapless::{pool, pool::singleton::Pool};

pool!(PDMPOOL: MaybeUninit<[u8; PDM_BUFFER_SIZE]>);
pool!(PCMPOOL: MaybeUninit<[f32; WINDOW_SIZE]>);

pub fn processors() -> (PdmProcessor, PostProcessor) {

    static mut PDM_STORAGE: [u8; PDM_BUFFER_SIZE * 32] = [0; PDM_BUFFER_SIZE * 32];
    static mut PCM_STORAGE: [u8; SAMPLE_STORAGE_SIZE] = [0; SAMPLE_STORAGE_SIZE];

    unsafe {
        PDMPOOL::grow(&mut PDM_STORAGE);
        PCMPOOL::grow(&mut PCM_STORAGE);
    }

    let (tx, rx) = unbounded();
    let pdm = PdmProcessor::new(tx);
    let post = PostProcessor::new(rx);
    (pdm, post)
}

pub struct PdmProcessor
{
    working_buffer: Vec<u8>,
    ready_buffer_tx: Sender<SampleBuffer<PCMPOOL, NUM_CHANNELS>>,
    pdm_processor: PdmProcessing<NUM_CHANNELS>,
    hooks: Vec<Box<dyn FnMut(&SampleBuffer<PCMPOOL, NUM_CHANNELS>) + Send>>
}

impl PdmProcessor
{
    pub fn new(tx: Sender<SampleBuffer<PCMPOOL, NUM_CHANNELS>>) -> Self {
        Self {
            working_buffer: Vec::new(),
            ready_buffer_tx: tx,
            pdm_processor: PdmProcessing::new(),
            hooks: Vec::new(),
        }
    }

    /// Collect chunks of pdm bytes into window size buffers and process them
    /// to RMS buffers.
    pub fn push_pdm_chunk(&mut self, packet: &[u8])
    {
        let bytes_remaining = PDM_BUFFER_SIZE - self.working_buffer.len();
        let mut offset = 0;
        if packet.len() >= bytes_remaining {
            // Enough bytes to complete this buffer. Finish filling it, and process it
            offset = bytes_remaining;
            self.working_buffer.extend_from_slice(&packet[0..bytes_remaining]);

            let mut pdm_data_box = PDMPOOL::alloc().unwrap().init(MaybeUninit::uninit());
            let pdm_data = unsafe { pdm_data_box.assume_init_mut() };
            pdm_data.copy_from_slice(&self.working_buffer);
            let pdm_buffer = PdmBuffer::new(0, pdm_data_box);
            let rms_buffer = compute_rms_buffer(pdm_buffer, &mut self.pdm_processor);
            let sample_buffer = compute_sample_buffer(rms_buffer, &mut self.pdm_processor);

            // Pass off to any hooks
            for cb in &mut self.hooks {
                cb(&sample_buffer);
            }

            // Queue for further processing
            self.ready_buffer_tx.send(sample_buffer).unwrap();

            // Create new buffer to continue filling
            self.working_buffer = Vec::with_capacity(PDM_BUFFER_SIZE);
        }

        // Store any bytes remaining after completing a buffer
        if offset < packet.len() {
            self.working_buffer.extend_from_slice(&packet[offset..]);
        }
    }

    pub fn add_hook(&mut self, hook: Box<dyn FnMut(&SampleBuffer<PCMPOOL, NUM_CHANNELS>) + Send>) {
        self.hooks.push(hook);
    }

}

const N_AZ_POINTS: usize = 100;
const IMAGE_GRID_RES: usize = 20;

pub struct PostProcessor {
    pub rms_series: Vec<f32>,
    pub latest_avg_spectrum: [f32; NFFT],
    buffer_rx: Receiver<SampleBuffer<PCMPOOL, NUM_CHANNELS>>,
    fft_processor: FftProcessor::<WINDOW_SIZE, NFFT>,
    az_beamformer: Box<BeamFormer<NUM_CHANNELS, N_AZ_POINTS, NFFT>>,
    image_beamformer: Box<BeamFormer<NUM_CHANNELS, {IMAGE_GRID_RES*IMAGE_GRID_RES}, NFFT>>,
}

pub unsafe fn unsafe_allocate<T>() -> Box<T> {
    let grid_box: Box<T>;

    use std::alloc::{alloc, Layout};
    let layout = Layout::new::<T>();
    let ptr = alloc(layout) as *mut T;
    grid_box = Box::from_raw(ptr);

    return grid_box;
}

impl PostProcessor {
    pub fn new(buffer_rx: Receiver<SampleBuffer<PCMPOOL, NUM_CHANNELS>>) -> Self {
        let az_focal_points = make_circular_focal_points::<N_AZ_POINTS>(1.0, 0.1);
        let image_focal_points = make_grid_focal_points::<IMAGE_GRID_RES, {IMAGE_GRID_RES*IMAGE_GRID_RES}>(1.0, 0.25);
        let mics = [
            [0.0, 0.05875, 0.],
            [0.050878992472335766, 0.029375000000000005, 0.],
            [-0.050878992472335766, -0.029375000000000005, 0.],
            [-0.050878992472335766, 0.029375000000000005, 0.],
            [7.1947999449907e-18, -0.05875, 0.],
            [0.05087899247233577, -0.029374999999999984, 0.],
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
        let mut az_beamformer = unsafe { unsafe_allocate::<BeamFormer<NUM_CHANNELS, N_AZ_POINTS, NFFT>>() };
        let mut image_beamformer = unsafe { unsafe_allocate::<BeamFormer<NUM_CHANNELS, {IMAGE_GRID_RES * IMAGE_GRID_RES}, NFFT>>() };
        // This causes stack overflow in debug builds
        //let mut az_beamformer = Box::new(BeamFormer::new());
        //let mut image_beamformer = Box::new(BeamFormer::new());

        az_beamformer.setup(mics, az_focal_points, FSAMPLE);
        image_beamformer.setup(mics, image_focal_points, FSAMPLE);

        Self {
            rms_series: vec![0.0; 200],
            latest_avg_spectrum: [0.0; NFFT],
            buffer_rx: buffer_rx,
            fft_processor: FftProcessor::new(),
            az_beamformer,
            image_beamformer,
        }
    }

    pub fn run(&mut self, start_freq: f32, end_freq: f32) -> ([f32; N_AZ_POINTS], [f32; IMAGE_GRID_RES * IMAGE_GRID_RES]) {
        let mut buf = self.buffer_rx.recv().unwrap();

        let mut rms: f32 = 0.0;
        for data in &buf.pcm {
            let data = data.as_ref().unwrap();
            let pcm = unsafe { data.assume_init_ref() };
            let (ch_rms, _mean) = dsp::pdm_processing::compute_rms_mean(pcm);
            rms += ch_rms;
        }

        rms /= NUM_CHANNELS as f32;

        // Store RMS to the time series used for display
        self.rms_series.remove(0);
        //self.rms_series.push(buf.rms);
        self.rms_series.push(buf.rms);

        let mut spectra = Spectra::blank();

        self.fft_processor.compute_ffts(&mut buf, &mut spectra);
        self.latest_avg_spectrum = spectra.avg_mag();

        let az_powers = self.az_beamformer.compute_power(&spectra, start_freq, end_freq);
        let image_powers = self.image_beamformer.compute_power(&spectra, start_freq, end_freq);

        (az_powers, image_powers)
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
