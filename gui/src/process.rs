
use std::rc::Rc;
use crossbeam_channel::{unbounded, Sender, Receiver};

use dsp::{
    buffer::PdmBuffer,
    cic::CicFilter,
};

use realfft::{
    RealFftPlanner,
    num_complex::Complex,
};

const FSAMPLE: f32 = 24e3;
const WINDOW_SIZE: usize = 1024;
const NUM_CHANNELS: usize = 6;
const DECIMATION: usize = 128;
const PDM_BUFFER_SIZE: usize  = WINDOW_SIZE * NUM_CHANNELS * DECIMATION / 8;
const DEC1: usize = 8; // First stage decimation ratio
const DEC2: usize = 4; // SEcond stage decimation ratio
const DEC3: usize = 4;
const ORDER1: usize = 4;
const ORDER2: usize = 4;
const ORDER3: usize = 4;
const SPEED_OF_SOUND: f32 = 343.0;

pub fn processors() -> (PdmProcessor, PostProcessor) {
    let (tx, rx) = unbounded();
    let pdm = PdmProcessor::new(tx);
    let post = PostProcessor::new(rx);
    (pdm, post)
}

pub struct PdmProcessor {
    working_buffer: Vec<[f32; NUM_CHANNELS]>,
    ready_buffer_tx: Sender<Vec<[f32; NUM_CHANNELS]>>,
    cic1: CicFilter::<DEC1, ORDER1, NUM_CHANNELS>,
    cic2: CicFilter::<DEC2, ORDER2, NUM_CHANNELS>,
    cic3: CicFilter::<DEC3, ORDER3, NUM_CHANNELS>,
}

impl PdmProcessor {
    pub fn new(tx: Sender<Vec<[f32; NUM_CHANNELS]>>) -> Self {
        Self {
            //rms_samples: vec![0.; 200],
            working_buffer: Vec::new(),
            ready_buffer_tx: tx,
            cic1: CicFilter::new(),
            cic2: CicFilter::new(),
            cic3: CicFilter::new(),
        }
    }

    pub fn push_pdm_chunk(&mut self, pdm: &[u8]) 
    {
        let samples_remaining = WINDOW_SIZE - self.working_buffer.len();
        let pdm_bytes_remaining = samples_remaining * DEC1 * DEC2 * NUM_CHANNELS;
        
        let cic1 = &mut self.cic1;
        let cic2 = &mut self.cic2;

        cic1.process_pdm_buffer(pdm, |frame1| {
            cic2.push_sample(frame1, |frame2| {
                self.cic3.push_sample(frame2, |frame3| {
                    let full_scale = usize::pow(DEC1, ORDER1 as u32) * usize::pow(DEC2, ORDER2 as u32) * usize::pow(DEC3, ORDER3 as u32);
                    let f32_frame = frame3.map(|x| x as f32 / full_scale as f32);
                    self.working_buffer.push(f32_frame);
                    if self.working_buffer.len() == WINDOW_SIZE {
                        let new_buf = std::mem::replace(&mut self.working_buffer, Vec::with_capacity(WINDOW_SIZE));
                        self.ready_buffer_tx.send(new_buf).unwrap();
                    }
                })
            })
        });
    }

    fn push_sample(&mut self, sample: [i32; NUM_CHANNELS]) {

    }
}


pub fn compute_rms(buf: &[[f32; NUM_CHANNELS]]) -> f32 {
    let length = buf.len();
    let mut mean = [0.0; NUM_CHANNELS];
    for frame in buf {
        for ch in 0..NUM_CHANNELS {
            mean[ch] += frame[ch];
        }
    }
    let mean = mean.map(|x| x / length as f32);
    let mut rms = [0.0; NUM_CHANNELS];
    for frame in buf {
        for ch in 0..NUM_CHANNELS {
            let x = frame[ch] - mean[ch];
            rms[ch] += x * x;
        }
    }
    rms = rms.map(|x| f32::sqrt(x / length as f32));
    let mut avg_rms = 0.0;
    for ch in 0..NUM_CHANNELS {
        avg_rms += rms[ch];
    }
    avg_rms /= NUM_CHANNELS as f32;
    20.0 * f32::log10(avg_rms)
}

pub fn compute_ffts(buf: &[[f32; NUM_CHANNELS]]) -> Vec<Vec<Complex<f32>>> {
    let mut fft_planner = RealFftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(WINDOW_SIZE);

    let mut spectra = Vec::with_capacity(NUM_CHANNELS);
    
    let mut mean = [0.0; NUM_CHANNELS];
    for frame in buf {
        for ch in 0..NUM_CHANNELS {
            mean[ch] += frame[ch];
        }
    }
    let mean = mean.map(|x| x / buf.len() as f32);

    for ch in 0..NUM_CHANNELS {
        
        let mut input: Vec<f32> = buf.iter().map(|frame| frame[ch] - mean[ch]).collect();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).unwrap();
        // Normalize magnitude
        let spectrum: Vec<Complex<f32>> = spectrum.iter().map(|x| x / WINDOW_SIZE as f32).collect();
        spectra.push(spectrum);
    }

    spectra
}

pub fn avg_fft_mag(spectra: &Vec<Vec<Complex<f32>>>) -> Vec<f32> {
    let mut avg_spectrum = vec![0.0; WINDOW_SIZE / 2];
    for ch in 0..spectra.len() {
        let s = &spectra[ch];
        for i in 0..WINDOW_SIZE/2 {
            avg_spectrum[i] += s[i].norm();
        }
    }
    for i in 0..WINDOW_SIZE/2 {
        avg_spectrum[i] = 20. * f32::log10(avg_spectrum[i] / spectra.len() as f32);
    }
    avg_spectrum
}

pub struct PostProcessor {
    pub rms_series: Vec<f32>,
    pub latest_avg_spectrum: Vec<f32>,
    buffer_rx: Receiver<Vec<[f32; NUM_CHANNELS]>>,
}

impl PostProcessor {
    pub fn new(buffer_rx: Receiver<Vec<[f32; NUM_CHANNELS]>>) -> Self {
        Self {
            rms_series: vec![0.0; 200],
            latest_avg_spectrum: vec![0.0; WINDOW_SIZE],
            buffer_rx: buffer_rx,
        }
    }

    pub fn run(&mut self) -> Vec<Vec<Complex<f32>>>{
        let buf = self.buffer_rx.recv().unwrap();
        
        let rms_db = compute_rms(&buf);
        self.rms_series.remove(0);
        self.rms_series.push(rms_db);

        
        let ffts = compute_ffts(&buf);
        let avg_fft = avg_fft_mag(&ffts);
        self.latest_avg_spectrum = avg_fft;
        ffts
    }
}

use ndarray::*;

pub fn make_circular_focal_points(n: usize, radius: f32, z: f32) -> Array2<f32> {
    let mut points = Array2::<f32>::from_elem((n, 3), 0.);

    for i in 0..n {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
        points[[i, 0]] = f32::sin(theta) * radius;
        points[[i, 1]] = f32::cos(theta) * radius;
        points[[i, 2]] = z;
    }
    points
}

pub fn compute_steering_vectors(focal_points: Array2<f32>, mic_positions: Array2<f32>, frequencies: &[f32]) -> Array3<Complex<f32>> {
    assert!(focal_points.shape()[1] == 3);
    assert!(mic_positions.shape()[1] == 3);

    let mut sv = Array3::<Complex<f32>>::zeros((focal_points.shape()[0], mic_positions.shape()[0], frequencies.len()));
    // Returns a focal_points x mic number x fft bin array of complex numbers
    for i in 0..focal_points.len_of(Axis(0)) {
        for j in 0..mic_positions.len_of(Axis(0)) {
            let fp = focal_points.row(i);
            let m = mic_positions.row(j);
            let dvec = &fp - &m;

            let d = f32::sqrt((&dvec * &dvec).sum());
            for k in 0..frequencies.len() {
                let f = frequencies[k];
                sv[[i, j, k]] = Complex::from_polar(1.0, std::f32::consts::PI * 2. * f * d / SPEED_OF_SOUND);
            }
        }
    }

    sv

}

pub struct BeamForm {
    freq_bin_start: usize,
    freq_bin_end: usize,
    steering_vectors: Array3::<Complex<f32>>,
}

impl BeamForm {
    pub fn new(f_start: f32, f_end: f32, focal_points: Array2<f32>, mic_positions: Array2<f32>) -> Self {
        assert!(f_end > f_start);
        assert!(focal_points.shape()[1] == 3);
        
        let freq_bin_start = f32::floor(f_start * WINDOW_SIZE as f32 / FSAMPLE) as usize;
        let freq_bin_end = f32::ceil(f_end * WINDOW_SIZE as f32 / FSAMPLE) as usize;
        let mut frequencies: Vec<f32> = Vec::new();
        for i in freq_bin_start..freq_bin_end+1 {
            frequencies.push(i as f32 * FSAMPLE / WINDOW_SIZE as f32);
        }
        println!("Bins: {} -> {}", freq_bin_start, freq_bin_end);
        println!("Freq: {:?}", &frequencies);
        let steering_vectors = compute_steering_vectors(focal_points, mic_positions, &frequencies);

       // println!("{:?}", steering_vectors);
        println!("{:?}", steering_vectors.dim());

        Self { freq_bin_start, freq_bin_end, steering_vectors }
    }

    pub fn compute_power(&self, spectra: Vec<Vec<Complex<f32>>>) -> Vec<f32> {
        let mut adjusted_spectra = Array3::<Complex<f32>>::zeros(self.steering_vectors.dim());
        for i in 0..adjusted_spectra.len_of(Axis(0)) {
            for j in 0..adjusted_spectra.len_of(Axis(1)) {
                for k in 0..adjusted_spectra.len_of(Axis(2)) {
                    adjusted_spectra[[i, j, k]] = self.steering_vectors[[i, j, k]] * spectra[j][k];
                }
            }
        }

        let n_mics = adjusted_spectra.len_of(Axis(1));
        // Sum channels axis and reduce to magnitude squared, then normalize by number of mics
        let summed = adjusted_spectra.sum_axis(Axis(1)).map(|x| x.norm() / n_mics as f32);
        // Average frequency bins for total power and convert to db
        let power: Vec<f32> = summed.mean_axis(Axis(1)).unwrap().map(|x| 20. * f32::log10(*x)).iter().cloned().collect();

        let mut max = -f32::INFINITY;
        let mut min = f32::INFINITY;
        for p in &power {
            if *p > max {
                max = *p;
            }
            if *p < min {
                min = *p;
            }
        }
        println!("Power range ({}, {})", min, max);
        //println!("{:?}", power);
        power
    }
}