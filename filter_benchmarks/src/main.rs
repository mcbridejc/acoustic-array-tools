#![no_std]
#![no_main]
#![feature(default_alloc_error_handler)]
#![feature(bench_black_box)]

mod logger;

use biquad::Coefficients;
use cortex_m_rt::{entry, exception};
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU32, Ordering};
use log::info;
use stm32h7xx_hal as hal;
use stm32h7xx_hal::{
    prelude::*,
    rcc::ResetEnable,
    device,
    device::interrupt,
};
use dasp::{Frame, Signal};
use dsp::{
    dasp::{BiquadFilter, CicDecimator, PdmSource},
    fir::{FirFilter}
};
use cmsis_dsp::filtering::{FloatFir, Q15Fir, Q31Fir};

/// TIME is an atomic u32 that counts milliseconds. Although not used
/// here, it is very useful to have for network protocols
static TIME: AtomicU32 = AtomicU32::new(0);

fn systick_init(syst: &mut device::SYST, clocks: hal::rcc::CoreClocks) {
    let c_ck_mhz = clocks.c_ck().to_MHz();

    let syst_calib = 0x3E8;

    syst.set_clock_source(cortex_m::peripheral::syst::SystClkSource::Core);
    syst.set_reload((syst_calib * c_ck_mhz) - 1);
    syst.enable_interrupt();
    syst.enable_counter();
}


use dsp::dasp::StreamingIterator;


// dasp links alloc, so it forces us to create an allocator. I don't expect to actually allocate anything though.
struct NullAllocator {}
 
unsafe impl core::alloc::GlobalAlloc for NullAllocator {
    unsafe fn alloc(&self, _layout: core::alloc::Layout) -> *mut u8 {
        core::ptr::null_mut() as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        panic!("dealloc cannot be called");
    }
}


#[global_allocator]
static fakealloc: NullAllocator = NullAllocator {};

/// And iterator to yield chunks of meaningless data that looks like pdm samples
struct BenchmarkFodder<'a, const CHUNK_SIZE: usize = 512> {
    buffer: [u8; CHUNK_SIZE],
    chunks_remaining: usize,
    _lifetime: core::marker::PhantomData<&'a ()>,
}

impl<'a, const CHUNK_SIZE: usize> BenchmarkFodder<'a, CHUNK_SIZE> {
    pub fn new(count: usize) -> Self {
        Self { buffer: [0u8; CHUNK_SIZE], chunks_remaining: count, _lifetime: core::marker::PhantomData }
    }
}

impl<'a, const CHUNK_SIZE: usize> StreamingIterator for BenchmarkFodder<'a, CHUNK_SIZE> 
{
    type Item = u8;

    fn get<'b>(&'b self) -> Option<&'b [Self::Item]> {
        if self.chunks_remaining > 0 {
            Some(core::hint::black_box(&self.buffer))
        } else {
            None
        }
    }

    fn next<'b>(&'b mut self) -> Option<&'b [Self::Item]> {
        self.advance();
        self.get()
    }

    fn advance<'b>(&'b mut self) {
        if self.chunks_remaining > 0 {
            self.chunks_remaining -= 1;
        }
    }
}


struct FrameIter {
    chunks_remaining: usize
}

impl FrameIter {
    pub fn new(count: usize) -> Self {
        Self{ chunks_remaining: count }
    }
}

impl Signal for FrameIter {
    type Frame = [i32; NUM_CHANNELS];

    fn next(&mut self) -> Self::Frame {
        if self.chunks_remaining > 0 {
            self.chunks_remaining -= 1;
        }
        core::hint::black_box([0i32; NUM_CHANNELS])
    }

    fn is_exhausted(&self) -> bool {
        self.chunks_remaining == 0
    }
}

const LOWPASS_COEFFS: [f32; 50] = [ 2.25311824e-04,  5.28281130e-03,  5.56600120e-03,  6.02832048e-03,
4.17572036e-03,  8.07054871e-05, -5.13611341e-03, -9.41126947e-03,
-1.04681413e-02, -6.85823472e-03,  1.06846141e-03,  1.08391130e-02,
1.83727875e-02,  1.94600250e-02,  1.16355776e-02, -4.21690294e-03,
-2.32930647e-02, -3.78186694e-02, -3.93643088e-02, -2.18421132e-02,
1.58325913e-02,  6.85293828e-02,  1.25715174e-01,  1.74201321e-01,
2.02001151e-01,  2.02001151e-01,  1.74201321e-01,  1.25715174e-01,
6.85293828e-02,  1.58325913e-02, -2.18421132e-02, -3.93643088e-02,
-3.78186694e-02, -2.32930647e-02, -4.21690294e-03,  1.16355776e-02,
1.94600250e-02,  1.83727875e-02,  1.08391130e-02,  1.06846141e-03,
-6.85823472e-03, -1.04681413e-02, -9.41126947e-03, -5.13611341e-03,
8.07054871e-05,  4.17572036e-03,  6.02832048e-03,  5.56600120e-03,
5.28281130e-03,  2.25311824e-04];


const CHUNKS_PER_BENCHMARK: usize = 4500;
const CHUNK_SIZE: usize = 512;
const NUM_CHANNELS: usize = 6;

const CMSIS_BUFFER_SIZE: usize = 96000 / 20;
const FIR_BATCH_SIZE:usize = 128;
const NUM_TAPS:usize = LOWPASS_COEFFS.len();

static mut FILTER_INPUT_BUFFER: [i32; CMSIS_BUFFER_SIZE] = [0; CMSIS_BUFFER_SIZE];
static mut FILTER_OUTPUT_BUFFER: [i32; CMSIS_BUFFER_SIZE] = [0; CMSIS_BUFFER_SIZE];
static mut FILTER_OUTPUT_BUFFER2: [f32; CMSIS_BUFFER_SIZE] = [0.; CMSIS_BUFFER_SIZE];

static filter_input_buffer_q15: [i16; CMSIS_BUFFER_SIZE] = [0; CMSIS_BUFFER_SIZE];
static mut filter_output_buffer_q15: [i16; CMSIS_BUFFER_SIZE] = [0; CMSIS_BUFFER_SIZE];

static mut PDM_INPUT_BUFFER: [u8; 3840] = [0; 3840];
static mut OUTPUT_BUFFER: [[i32; NUM_CHANNELS]; 1024] = [[0i32; NUM_CHANNELS]; 1024];

fn cmsis_fir_f32(input_buffer: &[f32], output_buffer: &mut [f32] )
{
    let mut filter_state = [0.0_f32; {NUM_TAPS + FIR_BATCH_SIZE}];
    let mut fir = FloatFir::<NUM_TAPS, FIR_BATCH_SIZE>::new(&LOWPASS_COEFFS, &mut filter_state);
    
    let output_buffer = core::hint::black_box(output_buffer);
    for _ in 0..60 {
        let mut pos: usize = 0;
        while pos < input_buffer.len() {
            let run_size = if input_buffer.len() - pos < FIR_BATCH_SIZE {
                input_buffer.len() - pos
            } else {
                FIR_BATCH_SIZE
            };
            fir.run(&input_buffer[pos..pos+run_size], &mut output_buffer[pos..pos+run_size]);
            pos += run_size;
        }
    }
}

use fixed::types::{I1F31, I1F15};

fn cmsis_fir_q15(input_buffer: &[I1F15], output_buffer: &mut [I1F15], coeffs: &[I1F15] )
{
    let mut filter_state = [I1F15::from_num(0); {NUM_TAPS + FIR_BATCH_SIZE}];
    let mut fir = Q15Fir::<NUM_TAPS, FIR_BATCH_SIZE>::new(coeffs, &mut filter_state);
    
    let output_buffer = core::hint::black_box(output_buffer);
    for _ in 0..60 {
        let mut pos: usize = 0;
        while pos < input_buffer.len() {
            let run_size = if input_buffer.len() - pos < FIR_BATCH_SIZE {
                input_buffer.len() - pos
            } else {
                FIR_BATCH_SIZE
            };
            fir.run(&input_buffer[pos..pos+run_size], &mut output_buffer[pos..pos+run_size]);
            pos += run_size;
        }
    }
}

fn cmsis_fir_q31(input_buffer: &[I1F31], output_buffer: &mut [I1F31], coeffs: &[I1F31] )
{
    let mut filter_state = [I1F31::from_num(0); {NUM_TAPS + FIR_BATCH_SIZE}];
    let mut fir = Q31Fir::<51, FIR_BATCH_SIZE>::new(&coeffs, &mut filter_state);
    
    let output_buffer = core::hint::black_box(output_buffer);
    for _ in 0..60 {
        let mut pos: usize = 0;
        while pos < input_buffer.len() {
            let run_size = if input_buffer.len() - pos < FIR_BATCH_SIZE {
                input_buffer.len() - pos
            } else {
                FIR_BATCH_SIZE
            };
            fir.run(&input_buffer[pos..pos+run_size], &mut output_buffer[pos..pos+run_size]);
            pos += run_size;
        }
    }
}

fn full_chain(output_buffer: &mut[[i32; NUM_CHANNELS]], hp_coeffs: Coefficients<f32>) {
    let mut pdm_source = BenchmarkFodder::<CHUNK_SIZE>::new(CHUNKS_PER_BENCHMARK);
    // Converts sequence of packets into sequence of N channel frames
    let frame_iter = dsp::dasp::FrameIterator::<NUM_CHANNELS, _>::new(&mut pdm_source);
    let pdm = PdmSource::new(dasp::signal::from_iter(frame_iter));
    let dec1 = CicDecimator::<_, 8, 4>::new(pdm);
    let dec2 = CicDecimator::<_, 4, 3>::new(dec1);
    let hp_filter = BiquadFilter::<_, NUM_CHANNELS>::new(dec2, hp_coeffs);
    let fir_filter = FirFilter::<_, NUM_TAPS, NUM_CHANNELS>::new(hp_filter, LOWPASS_COEFFS); 
    let mut dec3 = dsp::dasp::CicDecimator::<_, 4, 2>::new(fir_filter);
    let mut inptr = 0usize;
    while !dec3.is_exhausted() {
        output_buffer[inptr] = dec3.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
}

fn pdm_demux(output_buffer: &mut[[i32; NUM_CHANNELS]]) {
    let mut pdm_source = BenchmarkFodder::<CHUNK_SIZE>::new(CHUNKS_PER_BENCHMARK);
    let frame_iter = dsp::dasp::FrameIterator::<NUM_CHANNELS, _>::new(&mut pdm_source);
    let mut pdm = PdmSource::new(dasp::signal::from_iter(frame_iter));

    let mut inptr = 0usize;
    while !pdm.is_exhausted() {
        output_buffer[inptr] = pdm.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
}

fn pdm_cic_cic(output_buffer: &mut[[i32; NUM_CHANNELS]]) {
    let mut pdm_source = BenchmarkFodder::<CHUNK_SIZE>::new(CHUNKS_PER_BENCHMARK);
    let frame_iter = dsp::dasp::FrameIterator::<NUM_CHANNELS, _>::new(&mut pdm_source);
    let pdm = PdmSource::new(dasp::signal::from_iter(frame_iter));
    let dec1 = CicDecimator::<_, 4, 3>::new(pdm);
    let mut dec2 = CicDecimator::<_, 8, 4>::new(dec1);
    
    let mut inptr = 0usize;
    while !dec2.is_exhausted() {
        output_buffer[inptr] = dec2.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
}

fn fir_only(output_buffer: &mut[[i32; NUM_CHANNELS]]) {
    const FIR_SAMPLES: usize = 96000;

    let i32_source = FrameIter::new(FIR_SAMPLES);
    let mut fir_filter = FirFilter::<_, NUM_TAPS, NUM_CHANNELS>::new(i32_source, LOWPASS_COEFFS);
    
    let mut inptr = 0usize;
    while !fir_filter.is_exhausted() {
        output_buffer[inptr] = fir_filter.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
}

fn cic_i32(output_buffer: &mut[[i32; NUM_CHANNELS]]) {
    const SAMPLES: usize = 3_072_000;
    
    let i32_source = FrameIter::new(SAMPLES);
    let mut cic = CicDecimator::<_, 8, 4>::new(i32_source);
    
    let mut inptr = 0usize;
    while !cic.is_exhausted() {
        output_buffer[inptr] = cic.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
}

fn pdm_demux_batch(input_buffer: &[u8], output_buffer: &mut [[f32; NUM_CHANNELS]]) {
    for _ in 0..600 {
        let mut inptr = 0usize;
        dsp::cic::demux_pdm_all_channels(input_buffer, &mut |frame| {
            output_buffer[inptr] = frame;
            inptr = (inptr + 1) % output_buffer.len();
        });
    }

}

fn pdm_cic_batch(input_buffer: &[u8], output_buffer: &mut [[i32; NUM_CHANNELS]]) {
    let mut filter = dsp::cic::CicFilter::<8, 4, NUM_CHANNELS>::new();
    for _ in 0..600 {
        let mut inptr = 0usize;
        filter.process_pdm_buffer(input_buffer, &mut |frame| {
            output_buffer[inptr] = frame;
            inptr = (inptr + 1) % output_buffer.len();
        });
    }
}




fn benchmark<F>(func: F) 
where F: FnOnce()
{
    let dwt = unsafe { device::CorePeripherals::steal().DWT };
    let dwt_start = dwt.cyccnt.read();
    let syst_start = TIME.load(Ordering::Relaxed);
    func();
    let dwt_end = dwt.cyccnt.read();
    let syst_end = TIME.load(Ordering::Relaxed);
    let dwt_time = (dwt_end - dwt_start) as f32 / 400e3; // ms
    let syst_time = syst_end - syst_start; // ms

    log::info!("DWT: {}ms, SYST: {}", dwt_time, syst_time);
}


#[entry]
fn main() -> ! {
    let dp = device::Peripherals::take().unwrap();
    let mut cp = device::CorePeripherals::take().unwrap();
    
    logger::init();

    let pwr = dp.PWR.constrain().freeze();

    // Initialise SRAM3
    dp.RCC.ahb2enr.modify(|_, w| w.sram3en().set_bit());

    let rcc = dp.RCC.constrain();
    let ccdr = rcc
        .sys_ck(400.MHz())
        .hclk(200.MHz())
        .pll1_r_ck(100.MHz()) // for TRACECK
        .pll2_p_ck(18432.kHz()) // for SAI1
        .freeze(pwr, &dp.SYSCFG);

    // Initialise system...
    cp.SCB.enable_icache();
    // TODO: ETH DMA coherence issues
    cp.SCB.enable_dcache(&mut cp.CPUID);
    cp.DWT.enable_cycle_counter();

    // Initialise IO...
    let gpioa = dp.GPIOA.split(ccdr.peripheral.GPIOA);
    let gpiob = dp.GPIOB.split(ccdr.peripheral.GPIOB);
    let gpioc = dp.GPIOC.split(ccdr.peripheral.GPIOC);
    let gpioe = dp.GPIOE.split(ccdr.peripheral.GPIOE);
    let gpiof = dp.GPIOF.split(ccdr.peripheral.GPIOF);
    let gpiog = dp.GPIOG.split(ccdr.peripheral.GPIOG);
    let mut link_led = gpiob.pb0.into_push_pull_output(); // LED1, green
    link_led.set_low();

    assert_eq!(ccdr.clocks.hclk().raw(), 200_000_000); // HCLK 200MHz
    assert_eq!(ccdr.clocks.pclk1().raw(), 100_000_000); // PCLK 100MHz
    assert_eq!(ccdr.clocks.pclk2().raw(), 100_000_000); // PCLK 100MHz
    assert_eq!(ccdr.clocks.pclk4().raw(), 100_000_000); // PCLK 100MHz

    use biquad::*;

    const PDM_FREQ: u32 = 3_072_000;

    let hp_coeffs = Coefficients::<f32>::from_params(
        biquad::Type::HighPass,
        (PDM_FREQ / 32 as u32).hz(), // sample freq
        50u32.hz(), // cutoff freq
        biquad::Q_BUTTERWORTH_F32
    ).unwrap();


    let delay = cp.SYST.delay(ccdr.clocks);
    systick_init(&mut delay.free(), ccdr.clocks);
    unsafe {
        cp.SCB.shpr[15 - 4].write(128);
    } // systick exception priority

    let output_buffer = unsafe { OUTPUT_BUFFER.as_mut() };

    log::info!("Testing FIR implementation");
    // Generate some non zero input data
    let ibuf: &mut [f32; CMSIS_BUFFER_SIZE] = unsafe { core::mem::transmute(&mut FILTER_INPUT_BUFFER) };
    for i in 0..ibuf.len() {
        let sample = ibuf.get_mut(i).unwrap();
        let t: f32 = i as f32 / 24e3;
        *sample = 0.5 * cmsis_dsp::fast_math::FastMath::sin(2. * 3.14159 * 1e3 * t);
    }

    let out1: &mut [f32; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&mut FILTER_OUTPUT_BUFFER)) };
    let out2: &mut [f32; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&mut FILTER_OUTPUT_BUFFER2)) };

    let mut fir_filter = FirFilter::<_, NUM_TAPS, NUM_CHANNELS>::new(dasp::signal::from_iter(ibuf.iter().cloned()), LOWPASS_COEFFS);
    
    let mut inptr = 0usize;
    while !fir_filter.is_exhausted() {
        out1[inptr] = fir_filter.next();
        inptr += 1;
    }

    let mut state = [0.0f32; NUM_TAPS + FIR_BATCH_SIZE - 1];
    let mut cmsis_filt = cmsis_dsp::filtering::FloatFir::<NUM_TAPS, FIR_BATCH_SIZE>::new(&LOWPASS_COEFFS, &mut state);

    let mut pos: usize = 0;
    while pos < ibuf.len() {
        // cmsis_filt.run(&ibuf[pos..pos+1], &mut out2[pos..pos+1]);
        // pos += 1;
        let run_size = if ibuf.len() - pos < FIR_BATCH_SIZE {
            ibuf.len() - pos
        } else {
            FIR_BATCH_SIZE
        };
        cmsis_filt.run(&ibuf[pos..pos+run_size], &mut out2[pos..pos+run_size]);
        pos += run_size;
    }

    let mut error_count = 0;
    for i in 0..CMSIS_BUFFER_SIZE {
        if out1[i] != out2[i] {
            log::info!("Mismatch @ {}. {} / {}", i, out1[i], out2[i]);
            error_count += 1;
            if error_count > 100 {
                break;
            }
        }
    }


    // log::info!("Full chain");
    // benchmark( || { full_chain(output_buffer, hp_coeffs) });
    
    log::info!("Benchmarking FIR f32 with dasp implementation");
    benchmark(|| { fir_only(core::hint::black_box(output_buffer))} );

    log::info!("Benchmarking CMSIS-DSP FIR f32");
    let cmsis_input_buffer: &[f32; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&FILTER_INPUT_BUFFER)) };
    let cmsis_output_buffer: &mut [f32; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&mut FILTER_OUTPUT_BUFFER)) };
    benchmark(|| { cmsis_fir_f32(cmsis_input_buffer, cmsis_output_buffer) } );

    // log::info!("CMSIS FIR q31");
    // let cmsis_input_buffer: &[I1F31; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&filter_input_buffer)) };
    // let cmsis_output_buffer: &mut [I1F31; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&mut filter_output_buffer)) };
    // let q31_coeffs = LOWPASS_COEFFS.map(|x| I1F31::from_num(x) );
    // benchmark(|| { cmsis_fir_q31(cmsis_input_buffer, cmsis_output_buffer, &q31_coeffs) } );
    log::info!("Benchmarking CMSIS-DSP FIR q15");
    let cmsis_input_buffer_q15: &[I1F15; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&filter_input_buffer_q15)) };
    let cmsis_output_buffer_q15: &mut [I1F15; CMSIS_BUFFER_SIZE] = unsafe { core::hint::black_box(core::mem::transmute(&mut filter_output_buffer_q15)) };
    let q15_coeffs = LOWPASS_COEFFS.map(|x| I1F15::from_num(x) );

    benchmark(|| { cmsis_fir_q15(cmsis_input_buffer_q15, cmsis_output_buffer_q15, &q15_coeffs) } );


    log::info!("PDM demuxing");
    benchmark(|| { pdm_demux(core::hint::black_box(output_buffer))} );
    
    log::info!("CicDecimator + pdm demux");
    let mut pdm_source = BenchmarkFodder::<CHUNK_SIZE>::new(CHUNKS_PER_BENCHMARK);
    // Converts sequence of packets into sequence of N channel frames
    let frame_iter = dsp::dasp::FrameIterator::<NUM_CHANNELS, _>::new(&mut pdm_source);
    let pdm = PdmSource::new(dasp::signal::from_iter(frame_iter));
    let mut dec1 = CicDecimator::<_, 8, 4>::new(pdm);
    let dwt_start = cp.DWT.cyccnt.read();
    let mut inptr = 0usize;
    while !dec1.is_exhausted() {
        output_buffer[inptr] = dec1.next();
        inptr = (inptr + 1) % output_buffer.len();
    }
    let dwt_end = cp.DWT.cyccnt.read();
    let time = ((dwt_end - dwt_start) as f32) / 400e3;
    log::info!("Processed {} bytes in {} ({} - {})", CHUNK_SIZE * CHUNKS_PER_BENCHMARK, time, dwt_start, dwt_end);

    log::info!("CicDecimator + CicDeimator + pdm demux");
    benchmark(|| { pdm_cic_cic(core::hint::black_box(output_buffer)) } );
    
    log::info!("CicDecimator from i32 source");
    benchmark(|| { cic_i32(output_buffer) });

    log::info!("demux pdm buffer");
    let in_buffer = unsafe { PDM_INPUT_BUFFER.as_ref() };
    let out_buffer: &mut [[f32; NUM_CHANNELS]; 1024] = unsafe { core::mem::transmute(&mut OUTPUT_BUFFER) };
    benchmark(|| { pdm_demux_batch(core::hint::black_box(in_buffer), core::hint::black_box(out_buffer))});

    log::info!("pdm->cic");
    let out_buffer: &mut [[i32; NUM_CHANNELS]; 1024] = unsafe { core::mem::transmute(&mut OUTPUT_BUFFER) };
    benchmark(|| { pdm_cic_batch(core::hint::black_box(in_buffer), core::hint::black_box(out_buffer))});

    loop {
        let time = TIME.load(Ordering::Relaxed);
    }
}

#[exception]
fn SysTick() {
    TIME.fetch_add(1, Ordering::Relaxed);
}

#[exception]
unsafe fn HardFault(ef: &cortex_m_rt::ExceptionFrame) -> ! {
    panic!("HardFault at {:#?}", ef);
}

#[exception]
unsafe fn DefaultHandler(irqn: i16) {
    panic!("Unhandled exception (IRQn = {})", irqn);
}