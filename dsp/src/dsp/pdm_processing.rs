use crate::buffer::PdmBuffer;
use crate::buffer::RmsBuffer;

use crate::buffer::PoolSampleBuffer;
use crate::cic::CicFilter;
use crate::fir::FloatFirDecimate;

use core::cell::RefCell;
use core::mem::MaybeUninit;
use embassy_futures::yield_now;
use heapless::{
    pool::singleton::Pool,
    Vec,
};

/// Hard-coded FIR filter, designed for 96k sampling freq, with 8k cut-off, for
/// pre-decimation filter on last round of decimation
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

/// PdmProcessing struct stores filter state for decimation of all channels
const DEC1: usize = 8; // First stage decimation ratio
const DEC2: usize = 4; // Second stage decimation ratio
const DEC3: usize = 4; // Third stage decimation ratio
const ORDER1: usize = 4; // Order of first stage CIC
const ORDER2: usize = 3; // Order of second stage CIC

const SAMPLE_RATIO: usize = DEC1 * DEC2 * DEC3;

/// Number of samples at a time to pass to FIR filter Larger batch increases memory consumption, but
/// gives better performance up to a point
const BATCH_SIZE: usize = 64;

pub trait PdmProcessor {
    fn process_pdm(&self, pdm: &[u8], channel: usize, out: &mut[f32]);
}

struct PdmFilters {
    cic1: CicFilter::<DEC1, ORDER1>,
    cic2: CicFilter::<DEC2, ORDER2>,
    fir: FloatFirDecimate::<{LOWPASS_COEFFS.len()}, BATCH_SIZE, DEC3>,
    fir_buffer: [f32; BATCH_SIZE],
}

impl PdmFilters {
    pub fn new() -> Self {
        Self {
            cic1: CicFilter::new(),
            cic2: CicFilter::new(),
            fir: FloatFirDecimate::new(LOWPASS_COEFFS),
            fir_buffer: [0.0; BATCH_SIZE],
        }
    }
}

pub struct StaticPdmProcessor<const NCHAN: usize> {
    channels: Vec<RefCell<PdmFilters>, NCHAN>,
}

impl<const NCHAN: usize> StaticPdmProcessor<NCHAN> {
    pub fn new() -> Self {
        let mut cells: Vec<RefCell<PdmFilters>, NCHAN> = Vec::new();
        let mut i = 0;
        while i < NCHAN {
            cells.push(RefCell::new(PdmFilters::new())).ok().expect("Overran heapless vec");
            i += 1;
        }
        Self {
            channels: cells
        }
    }
}

impl<const NCHAN: usize> PdmProcessor for StaticPdmProcessor<NCHAN> {
    fn process_pdm(&self, pdm: &[u8], channel: usize, out: &mut [f32])
    {
        let total_samples = pdm.len() * 8 / NCHAN / SAMPLE_RATIO;
        let mut skip_samples = total_samples - out.len();
        let mut outpos: usize = 0;
        let mut batchpos: usize = 0;
        //  Deref to get around the RefMut and get a real reference
        // https://stackoverflow.com/questions/47060266/error-while-trying-to-borrow-2-fields-from-a-struct-wrapped-in-refcell
        let f = &mut *self.channels[channel].borrow_mut();
        let cic1 = &mut f.cic1;
        let cic2 = &mut f.cic2;
        let fir = &mut f.fir;
        let fir_buffer = &mut f.fir_buffer;
        cic1.process_pdm_buffer::<_, NCHAN>(channel, pdm, |sample1| {
            cic2.push_sample(sample1, |sample2| {
                // Convert to float
                // Scale so that full-scale input results in +/- 1.0 output
                const FULL_SCALE: usize = usize::pow(DEC1, ORDER1 as u32) * usize::pow(DEC2, ORDER2 as u32);
                let float_sample = sample2 as f32 / FULL_SCALE as f32;

                fir_buffer[batchpos] = float_sample;
                batchpos += 1;
                if batchpos == fir_buffer.len() {
                    // Low pass
                    fir.process_block(fir_buffer, &mut out[outpos..outpos + BATCH_SIZE / DEC3]);
                    outpos += BATCH_SIZE / DEC3;

                    if skip_samples > 0 {
                        if skip_samples >= BATCH_SIZE / DEC3 {
                            // just skip the whole batch
                            outpos = 0;
                            skip_samples -= BATCH_SIZE / DEC3;
                        } else {
                            // We have to shift the data
                            for i in 0..BATCH_SIZE / DEC3 - skip_samples {
                                out[i] = out[i+skip_samples];
                            }
                            outpos -= skip_samples;
                            skip_samples = 0;
                        }
                    }
                    batchpos = 0;
                }
            });
        });

        // Finish any remaining partial batch
        if batchpos > 0 {
            fir.process_block(&fir_buffer[0..batchpos], &mut out[outpos..outpos + batchpos / DEC3]);
        }
    }
}

/// Return RMS of signal in dbFS, and the calculated DC offset
pub fn compute_rms_mean(buf: &[f32]) -> (f32, f32) {
    let length = buf.len();
    let mean = compute_mean(buf);
    let mut rms = 0.0f32;
    for sample in buf {
        let x = sample - mean;
        rms += x * x;
    }
    rms = libm::sqrtf(rms / length as f32);
    // Convert to db relative to full scale
    // i.e. 20 * log10(rms * sqrt(2)). Note: 20*log10(sqrt(2)) ~= 3.0103
    // This means a sine wave with +/-1.0 amplitude is 0db.
    let db_fs = 20.0 * libm::log10f(rms) + 3.0103;
    (db_fs, mean)
}

pub fn compute_mean(buf: &[f32]) -> f32 {
    let mut mean: f32 = 0.0;
    for sample in buf {
        mean += sample;
    }
    mean / buf.len() as f32
}

/// First transformation of PdmBuffer with raw captured data to RmsBuffer with raw PDM plus PCM
/// conversion for one channel and associated statistics which can be used to decide which segments
/// to perform full processing on
pub fn compute_rms_buffer<D, C, P, const N: usize, const M: usize>(pdm: PdmBuffer<D>, processor: &P) -> RmsBuffer<D, C>
where
    D: Pool<Data = MaybeUninit<[u8; M]>>,
    C: Pool<Data = MaybeUninit<[f32; N]>>,
    P: PdmProcessor + ?Sized,
{
    // Allocate a new buffer for the output

    // TODO: Think more about failing here Ultimately, it
    // should not fail here, and if it is going to we should dump unprocessed buffers in the queue
    let out_data_box = C::alloc().unwrap();
    let mut out_data_box = out_data_box.init(core::mem::MaybeUninit::uninit());
    let out_data = unsafe { out_data_box.assume_init_mut() };
    //processor.process_pdm(pdm, 0, out_data.)
    let pdm_bytes = unsafe { pdm.pdm_data.assume_init_ref() };
    processor.process_pdm(pdm_bytes, 0, out_data);

    let (rms, mean) = compute_rms_mean(out_data);
    for i in 0..out_data.len() {
        out_data[i] -= mean;
    }

    RmsBuffer {
        pdm_data: Some(pdm.pdm_data),
        ch1_pcm: Some(out_data_box),
        index: pdm.index,
        rms,
    }
}

/// Second transformation of RmsBuffer to SampleBuffer
/// All channels are converted to PCM and PDM data is freed.
pub async fn compute_sample_buffer<
    D,
    C,
    P,
    const N: usize,
    const M: usize,
    const NCHAN: usize,
    >(
        mut rms: RmsBuffer<D, C>,
        processor: &P
    ) -> PoolSampleBuffer<C, NCHAN>
where
    D: Pool<Data = MaybeUninit<[u8; M]>>,
    C: Pool<Data= MaybeUninit<[f32; N]>>,
    P: PdmProcessor + ?Sized,
{
    let pdm_data_box = rms.pdm_data.unwrap();
    let pdm_data = unsafe { pdm_data_box.assume_init_ref() };

    let mut result = PoolSampleBuffer::new();

    // Ch0 is already processed!
    result.pcm[0] = rms.ch1_pcm.take();
    result.rms = rms.rms;
    result.index = rms.index;

    for ch in 1..NCHAN {
        // It is assumed that allocation can't fail here, something which can be guaranteed by
        // design if the application keeps at most one PCM buffer per RmsBuffer -- the number of
        // these is limited by the number slots in the PDM pool -- plus one more for each MIC chan
        // and allocates that many PCM buffers (N_PDM_BUFFERS + N_CHAN).
        let mut out_data_box = C::alloc().unwrap().init(MaybeUninit::uninit());
        let out_data = unsafe { out_data_box.assume_init_mut() };

        processor.process_pdm(pdm_data, ch, out_data);

        let mean = compute_mean(out_data);
        for i in 0..out_data.len() {
            out_data[i] -= mean;
        }
        result.pcm[ch] = Some(out_data_box);

        // Yield between channels to avoid blocking other processing tasks for too long
        yield_now().await;
    }

    result
}