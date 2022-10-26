use heapless::{
    pool, 
    pool::singleton::{Box, Pool}
};
use heapless::spsc::Queue;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

use dsp::{
    azimuth,
    beamforming::SmallMemBeamFormer,
    buffer::{PdmBuffer, StaticSpectra, Spectra},
    fft::{
        fftimpl::Fft,
    },
    pdm_processing::StaticPdmProcessor,
    pipeline,
    pipeline::{ProcessingQueue},
};
use crate::{device, dma};
use crate::device::interrupt;
use crate::pdm::PdmInput;
use crate::hal::dma::traits::TargetAddress;
use log::info;


pub fn make_circular_focal_points<const N: usize>(radius: f32, z: f32) -> [[f32; 3]; N] {
    let mut points = [[0.0; 3]; N];

    let mut i = 0;
    while i < N {
        let theta = 2.0 * core::f32::consts::PI * i as f32 / N as f32;
        points[i][0] = libm::sinf(theta) * radius;
        points[i][1] = libm::cosf(theta) * radius;
        points[i][2] = z;
        i += 1;
    }
    points
}

// Flag to be set by DMA IRQ if it does not execute, or cannot aquire a free buffer in time
pub static mut DMA_OVERRUN: AtomicBool = AtomicBool::new(false);

// Create two memory pools, which will consume the bulk of AXISRAM + SRAM1 + SRAM2
// One (PDMPOOL) is used for raw PDM buffers, and shall be large enough to hold the PDM data required for 
// one sample window. The other (PCMPOOL) holds a single channel of PCM samples for one sample window.
// It's important to get the sizing right, and it's a little tricky because the memory has to be dispersed over the 
// different region, and we have to account for the fact that the heapless pool may require some overhead data.
pool!(PDMPOOL: MaybeUninit<[u8; PDM_BUFFER_SIZE]>);
pool!(PCMPOOL: MaybeUninit<[f32; WINDOW_SIZE]>);

/// The sample frequency after PCM conversion
const FSAMPLE: f32 = 24e3;
/// The ratio of PDM frequency to PCM frequency
const DECIMATION: usize = 128;
/// The number of input channels
const NCHAN: usize = 6;
/// The number of PCM samples in each processing window -- i.e. the size of input to FFTs
const WINDOW_SIZE: usize = 1024;
/// The number of complex samples in a spectrum after FFT
const NFFT: usize = WINDOW_SIZE / 2 + 1;
/// The number of "extra" PCM samples to be produced in order to prime filters
/// This is necessary because when we drop (i.e. don't do PDM to PCM processing) sample windows
/// due to lack of CPU time, this introduces discontinuities which creates noise. This needs to be
/// long enough to flush the discontinuity out of the PDM filters.
const PRELUDE_SAMPLES: usize = 16;
/// The number of bytes in each PDM buffer 
const PDM_BUFFER_SIZE: usize = NCHAN * (WINDOW_SIZE + PRELUDE_SAMPLES) * DECIMATION / 8;

/// The expected number of PDM buffers to allocate
const NUM_PDM_BUFFERS: usize = 7;

/// Size in bytes of one PCMPOOL buffer; enough to hold a window for a single channel
const PCM_BUFFER_SIZE: usize = 4 * WINDOW_SIZE;
/// The number of buffers required in the PCM POOL
/// This this is simple: we need one buffer for each PDM buffer (to store the ch1 data calculated to determine RMS), 
/// plus enough to finish processing a single frame (i.e. to hold the NCHAN - 1 for that frame)
const NUM_PCM_BUFFERS: usize = NUM_PDM_BUFFERS + NCHAN - 1;
/// The number of bytes required for a single PCM buffer

// The entire AXIBUF will be used for PDM buffers
// This gets us 5 buffers (using window size of 1024 + PRELUDE_SAMPLES = 13 )
const AXIBUF_SIZE: usize = 512 * 1024;
#[link_section = ".axisram.pdm"]
static mut AXIBUF: [u8; AXIBUF_SIZE] = [0; AXIBUF_SIZE];

// SRAM 1 and 2 are contiguous, and combined in the linker script as one region. SRAM3 is only 32K,
// and some is needed by ethernet anyway, so we don't bother with it. 

// Allocate enough of the SRAM12 for two more PDM buffer
// Empirically determined we need a few extra bytes for heapless pool
const PDM_ALLOC2: usize = 2 * PDM_BUFFER_SIZE + 8;
#[link_section = ".sram12.pdm"]
static mut SRAMBUF1: [u8; PDM_ALLOC2] = [0; PDM_ALLOC2];


// Allocate storage for the PCM buffers
const PCM_ALLOC1: usize = (PCM_BUFFER_SIZE + 4) * NUM_PCM_BUFFERS;
#[link_section = ".sram12.pdm"]
static mut SRAMBUF2: [u8; PCM_ALLOC1] = [0; PCM_ALLOC1];

/// Positions of the microphones for each input channel
const MICS: [[f32; 3]; NCHAN] = [
    [-0.05088, 0.0294, 0.],
    [-0.05088, -0.0294, 0.],
    [0.05088, 0.0294, 0.],
    [0.0, 0.05875, 0.],
    [0.05088, -0.0294, 0.],
    [0., -0.05875, 0.],
];
/// The number of radial focal points to compute
const NFOCAL: usize = 60;
/// The distance of each radial focal point from origin
const FOCAL_RADIUS: f32 = 1.0;
/// The z position of the radial focal points (i.e. height above mic plane)
const FOCAL_Z: f32 = 0.1;
/// The low-end of the frequency range to be used for beamforming
const START_FREQ: f32 = 500.0;
/// The upper-end of the frequency range to be used for beamforming
const END_FREQ: f32 = 2500.0;



// Up to two PDM pool buffers can be allocated for the DMA at a given time.
static mut SLOT0_BUF: Option<Box<PDMPOOL>> = None;
static mut SLOT1_BUF: Option<Box<PDMPOOL>> = None;

static mut DMA_STREAM: MaybeUninit<dma::DmaStream<0>> = MaybeUninit::uninit();

/// Queue to hold raw PDM data recevied from DMA. This is allocating small RAM to hold references;
/// the main data is stored in pools from PDMPOOL
static mut READY_PDM_QUEUE: Queue<PdmBuffer<PDMPOOL>, 16> = Queue::new();
/// Queue for holding partially processed RMS buffers, while they await full processing
/// This queue allows for pruning data buffers based on RMS value, removing the lowest power blocks
/// first. When a buffer is dropped, the PDM and PCM buffers are returned to the pool, but the RmsBuffer
/// object is kept as a placeholder.
static mut RMS_BUFFER_QUEUE: ProcessingQueue<PDMPOOL, PCMPOOL, 128> = ProcessingQueue::new();
/// Holds filter state for PDM to PCM conversion for all channels
static mut PDM_PROCESSOR: Option<StaticPdmProcessor<NCHAN>> = None; 

/// Holds some setup state for FFT computation
static mut FFT_PROCESSOR: Option<Fft<WINDOW_SIZE>> = None;
/// Storage for FFT data for all channels, for a single frame
static mut SPECTRA_BUFFER: StaticSpectra<{WINDOW_SIZE / 2 + 1}, NCHAN> = StaticSpectra::blank();
/// Storage for beamformer data, i.e. pre-computed steering vectors
static mut BEAMFORMER: SmallMemBeamFormer<NCHAN, NFOCAL> = SmallMemBeamFormer::new();

/// Minimum RMS value to be considered in azimuth filter
const RMS_THRESH: f32 = -55.0;
/// Decay constant which controls the ramp rate of post-event elevated RMS threshold in az filter
const RMS_DECAY: f32 = 0.25;
/// Allocate storage for az filter state
static mut AZFILTER: azimuth::AzFilter = azimuth::AzFilter::new(RMS_THRESH, RMS_DECAY);

pub struct AudioReader {
    pdm: Option<PdmInput>
}

// Just a way to enforce a singleton, so only one AudioReader can be created.
static mut THE_ONE_TRUE_READER: Option<AudioReader> = Some(AudioReader { pdm: None });

impl AudioReader {
    pub fn init(
        sai1: &'static device::sai4::RegisterBlock,
        dma1: &'static device::dma1::RegisterBlock,
    ) -> AudioReader {
        let reader = unsafe { THE_ONE_TRUE_READER.take() };
        // Panic if necessary, before re-initializing
        let mut reader = reader.unwrap();

        // Allocate buffer pools, and panic if pool ends up allocating less than the expected number of 
        // blocks due to alignment losses or any other reason. 
        let mut pdm_bufs: usize = 0;
        let mut pcm_bufs: usize = 0;
        pdm_bufs += unsafe { PDMPOOL::grow(&mut AXIBUF) };
        pdm_bufs += unsafe { PDMPOOL::grow(&mut SRAMBUF1) };
        pcm_bufs += unsafe { PCMPOOL::grow(&mut SRAMBUF2) };

        assert!(pdm_bufs == NUM_PDM_BUFFERS, "Only got {} pdm blocks", pdm_bufs);
        assert!(pcm_bufs == NUM_PCM_BUFFERS, "Only got {} pcm blocks", pcm_bufs);

        unsafe { FFT_PROCESSOR = Some(Fft::new()) };
        unsafe { PDM_PROCESSOR = Some(StaticPdmProcessor::new()) };

        // Steering vectors have to be calculated once at startup (although I suppose it's probably
        // possible to make this a const function...)
        let beamformer = unsafe { &mut BEAMFORMER };
        // I'd love for this to be const, but trig functions aren't const so I guess it can just go
        // on the stack
        let focal_points = make_circular_focal_points::<NFOCAL>(FOCAL_RADIUS, FOCAL_Z);
        beamformer.setup(MICS, focal_points, FSAMPLE);

        let mut pdm = PdmInput::new(sai1);
        // Set DMA mux for DMA1 channel 0 to request 87, which is the SAI1 periph.
        dma::set_dma_request_mux(0, 87);

        // Prepare the DMA for the first transfer. From here on, the IRQ handler will manage the DMA.
        let dmastream = dma::DmaStream::<0>::new(dma1);
        dmastream.set_psize(dma::DataSize::HalfWord);
        // Allocate the first two PDM buffer. This allocation can't fail -- one hopes! -- because it's
        // the first.
        let mut pdm_data_box1 = PDMPOOL::alloc().unwrap().init(MaybeUninit::uninit());
        let mut pdm_data_box2 = PDMPOOL::alloc().unwrap().init(MaybeUninit::uninit());
        // Setup the first transfer
        let xfer_ptr1 = pdm_data_box1.as_mut_ptr();
        let xfer_ptr2 = pdm_data_box2.as_mut_ptr();
        dmastream.start_p2m_transfer(pdm.address(), xfer_ptr1, Some(xfer_ptr2), PDM_BUFFER_SIZE);

        // Store the allocated buffers and DMA stream in their static slots for IRQ
        unsafe { SLOT0_BUF = Some(pdm_data_box1) };
        unsafe { SLOT1_BUF = Some(pdm_data_box2) };
        dmastream.enable_interrupt();
        unsafe { DMA_STREAM.write(dmastream) };

        // Setup PDM input from the SAI1 peripheral.
        pdm.init();
        pdm.enable_dma();
        
        reader.pdm = Some(pdm);
        reader
    }

    // /// Check if a filled buffer is available and return it if so
    // /// Buffers must be returned via `return_buffer`.
    // pub fn poll_for_data(&self) -> Option<Buffer> {
    //     let mut rts_consumer = unsafe { READY_TO_SEND.split().1 };
    //     rts_consumer.dequeue()
    // }

    // /// Return a buffer previously received via `poll_for_data`
    // pub fn return_buffer(&self, buf: Buffer) {
    //     let mut rtf_producer = unsafe { READY_TO_FILL.split().0 };
    //     rtf_producer.enqueue(buf).unwrap();
    // }

    pub async fn preprocess(&mut self) {
        let pdm_processor = unsafe { PDM_PROCESSOR.as_ref().unwrap() };
        let mut in_consumer = unsafe { READY_PDM_QUEUE.split().1 };
        let out_queue = unsafe { &RMS_BUFFER_QUEUE };
        return pipeline::preprocess(pdm_processor, &mut in_consumer, out_queue, RMS_THRESH).await;
    }

    pub async fn postprocess(&mut self) -> Option<f32> {
        let in_queue = unsafe { &RMS_BUFFER_QUEUE };
        let pdm_processor = unsafe { PDM_PROCESSOR.as_ref().unwrap() };
        let spectra = unsafe { &mut SPECTRA_BUFFER };
        let fft_processor = unsafe { FFT_PROCESSOR.as_mut().unwrap() };
        let az_filter = unsafe { &mut AZFILTER };

        let new_spectra = pipeline::process_spectra::<_, _, _, _, _, NCHAN, NFFT>(pdm_processor, fft_processor, in_queue, spectra).await;
        if new_spectra {
            let moment = if spectra.data_valid() {
                let beamformer = unsafe { &BEAMFORMER };
                let mut power_out = [0.0; NFOCAL];
                pipeline::process_beamforming(beamformer, spectra, &mut power_out, START_FREQ, END_FREQ);
                info!("X {}", spectra.rms());
                //info!("{:?}", &spectra.as_slice(0).unwrap()[0..20]);
                Some(azimuth::weighted_azimuth(&power_out))
                
            } else {
                //info!("O {}", spectra.rms());
                None
            };
            az_filter.push(moment, spectra.rms())
        } else {
            None
        }

    }
}

// IRQ goes off on completion of a transfer. We then swap out the inactive buffer
// for a for a fresh one, and pass the completed one off to the ready-to-send 
// queue for transmission
#[interrupt]
fn DMA1_STR0() {
    static mut FIRST_BUFFER_COMPLETE: bool = true;
    static mut BUFFER_INDEX: u32 = 0;

    let stream = unsafe {DMA_STREAM.assume_init_mut()};
    stream.clear_interrupts();
    let mut out_producer = unsafe { READY_PDM_QUEUE.split().0 };

    //info!("DMA");
    let next_buf = PDMPOOL::alloc();

    // let mut rtf_consumer = unsafe { READY_TO_FILL.split().1 };
    // let mut rts_producer = unsafe { READY_TO_SEND.split().0 };
    // let next_buf = rtf_consumer.peek();

    // If no new buffer is available, just let the DMA keep going with the buffers it has.
    // Hopefully, this does not happen because buffers are dropped from the processing queue
    // before we ever reach this point.
    if next_buf.is_none() {
        info!("OVERRUN");
        unsafe { DMA_OVERRUN.store(true, Ordering::Relaxed); }
        return;
    }
    let mut next_buf = next_buf.unwrap().init(MaybeUninit::uninit());

    let dma_first_buffer_complete = stream.get_current_target() == dma::TargetBuffer::Buffer1;
    // If the DMA somehow got ahead of us, flag it as an overrun and take no action.
    // This really shouldn't happen, because the transfers are long and we have an 
    // entire transfer window to service the IRQ.
    if *FIRST_BUFFER_COMPLETE != dma_first_buffer_complete {
        info!("DMA SYNC ERROR");
        unsafe { DMA_OVERRUN.store(true, Ordering::Relaxed); }
        return
    }

    let completed_buffer = if *FIRST_BUFFER_COMPLETE {
        unsafe { SLOT0_BUF.take().unwrap() }
    } else {
        unsafe { SLOT1_BUF.take().unwrap() }
    };

    let pdm_buffer = PdmBuffer::new(*BUFFER_INDEX, completed_buffer);
    *BUFFER_INDEX += 1;
    // As long as we make the READY_PDM_QUEUE size >= NUM_PDM_BUFFERS, it should not be possible for this to fail
    out_producer.enqueue(pdm_buffer).ok().expect("failed queuing pdm buffer");

    // Trigger the pre-processing task
    let mut nvic = unsafe { crate::device::CorePeripherals::steal().NVIC };
    nvic.request(crate::device::Interrupt::LPTIM5);

    //info!("Q: {}", out_producer.len());
    
    if *FIRST_BUFFER_COMPLETE {
        stream.load_memory0(next_buf.as_mut_ptr());
        unsafe { SLOT0_BUF = Some(next_buf) };
        *FIRST_BUFFER_COMPLETE = false;
    } else {
        stream.load_memory1(next_buf.as_mut_ptr());
        unsafe { SLOT1_BUF = Some(next_buf) };
        *FIRST_BUFFER_COMPLETE = true;
    }
}