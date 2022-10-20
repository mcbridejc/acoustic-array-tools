use core::cell::RefCell;
use core::mem::MaybeUninit;
use embassy_futures::yield_now;
use heapless::Vec;
use heapless::pool::singleton::{Pool};
use crate::{
    beamforming::{BeamFormer, FftProcessor},
    buffer::{PdmBuffer, RmsBuffer, Spectra},
    pdm_processing,
    pdm_processing::{compute_rms_buffer, PdmProcessor},
};
use crossbeam::channel::Receiver;

pub struct ProcessingQueue<D, C, const QSIZE: usize>
where
    D: Pool,
    C: Pool,
{
    queue: Vec<RmsBuffer<D, C>, QSIZE>,
}

impl<D, C, const QSIZE: usize> ProcessingQueue<D, C, QSIZE>
where
    D: Pool,
    C: Pool,
{
    pub fn new() -> Self {
        Self {
            queue: Vec::new()
        }
    }

    pub fn push(&mut self, pdm_buffer: RmsBuffer<D, C>) -> Result<(), RmsBuffer<D, C>> {
        self.queue.push(pdm_buffer)
    }

    /// Total number of buffers in the queue
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Number of buffers containing valid PDM data -- i.e. buffers which have not been dropped
    pub fn valid(&self) -> usize {
        let mut count = 0;
        for buf in &self.queue {
            if buf.pdm_data.is_some() {
                count += 1;
            }
        }
        count
    }

    /// Push the new buffer into the queue, and if necessary drop data from as many buffers as
    /// needed to reduce the number of valid buffers in the queue to max_len. Buffers are dropped
    /// according to rms, with the lowest values dropped virst. The buffer currently being added is
    /// eligible for dropping.
    pub fn push_with_limit(&mut self, pdm_buffer: RmsBuffer<D, C>, max_len: usize) -> Result<(), RmsBuffer<D, C>> {
        let result = self.push(pdm_buffer);

        if result.is_ok() {
            let mut valid = self.valid();

            while valid > max_len {
                self.prune();
                valid -= 1;
            }            
        }

        result
    }

    /// Remove the the buffer with the lowest RMS value from the queue
    pub fn prune(&mut self) {
        if self.queue.len() == 0 {
            return;
        }

        let mut min_idx = 0;
        let mut min_value = f32::INFINITY;
        for i in 0..self.queue.len() {
            let buf = &self.queue[i];
            if buf.pdm_data.is_some() && buf.rms < min_value {
                min_value = buf.rms;
                min_idx = i;
            }
        }
        let pruned = &mut self.queue[min_idx];
        // Data blocks are dropped and will be returned to their respective pools
        pruned.pdm_data.take();
        pruned.ch1_pcm.take();
    }

    pub fn pop(&mut self) -> Option<RmsBuffer<D, C>> {
        if self.queue.len() > 0 {
            Some(self.queue.remove(0))
        } else {
            None
        }
    }
}


pub async fn preprocess<D, C, const N: usize, const PDMSIZE: usize, const PCMSIZE: usize>(
    processor: &dyn PdmProcessor,
    rx: &mut Receiver<PdmBuffer<D>>,
    tx: &RefCell<ProcessingQueue<D, C, N>>,
    rms_threshold: f32
)
where
    D: Pool<Data = MaybeUninit<[u8; PDMSIZE]>>,
    C: Pool<Data = MaybeUninit<[f32; PCMSIZE]>>,
{
    while let Ok(pdm) = rx.try_recv() {
        let mut rms: RmsBuffer<D, C> = compute_rms_buffer(pdm, processor);
        if rms.rms > rms_threshold {
            tx.borrow_mut().push_with_limit(rms, 4).ok();
        } else {
            // Free the data, and push into the queue. No limiting necessary, because limiting only
            // applies to active buffers (i.e. buffers which still have data to process)
            rms.ch1_pcm.take();
            rms.pdm_data.take();
            tx.borrow_mut().push(rms).ok();
        }
    }
    yield_now().await;
}

pub async fn process_spectra<
    D,
    C,
    const N: usize,
    const PDMSIZE: usize,
    const WINDOW_SIZE: usize,
    const NCHAN: usize,
    const NFFT: usize,
> (
    pdm_processor: &dyn PdmProcessor,
    fft_processor: &mut FftProcessor,
    rx: &RefCell<ProcessingQueue<D, C, N>>,
    spectra: &mut dyn Spectra,
) -> bool
where
    D: Pool<Data = MaybeUninit<[u8; PDMSIZE]>>,
    C: Pool<Data = MaybeUninit<[f32; WINDOW_SIZE]>>,
{
    // Finish PDM-to-PCM converion on remaining channels

    let rms = rx.borrow_mut().pop();
    if let Some(rms) = rms {
        if rms.pdm_data.is_none() {
            // Return an empty sample buffer; no data but pass on the RMS value and index as
            // placeholder
            spectra.set_rms(rms.rms);
            spectra.set_index(rms.index);
            return true;
        } else {
            // Processes all PDM channels to PCM, and frees the PDM buffers
            let mut sample_buf = pdm_processing::compute_sample_buffer::<_, _, _, WINDOW_SIZE, PDMSIZE, NCHAN>(rms, pdm_processor).await;
            // Take FFTs
            fft_processor.compute_ffts(&mut sample_buf, spectra).await;
            spectra.set_rms(sample_buf.rms);
            spectra.set_index(sample_buf.index);
            true
        }
    } else {
        // No blocks ready for processing
        false
    }
}

pub fn process_beamforming<
    const WINDOW_SIZE: usize,
    const NFOCAL: usize,
    const NFFT: usize,
    const NCHAN: usize
> (
    beamformer: &dyn BeamFormer,
    spectra: &dyn Spectra,
    power_out: &mut [f32; NFOCAL],
    start_freq: f32,
    end_freq: f32
) {
    beamformer.compute_power(spectra, power_out, start_freq, end_freq)
}


