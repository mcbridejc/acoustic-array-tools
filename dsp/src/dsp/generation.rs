
use crate::buffer::SampleBuffer;

pub struct HeapSampleBuffer {
    samples: Vec<Vec<f32>>,
}

impl HeapSampleBuffer {
    pub fn new(nchan: usize, window_size: usize) -> Self {
        let mut samples = Vec::new();
        if nchan == 0 {
            panic!("Cannot create a HeapSampleBuffer with 0 channels")
        }
        for _ in 0..nchan {
            samples.push(vec![0.0; window_size]);
        }
        Self { samples }
    }
}

impl SampleBuffer for HeapSampleBuffer {
    fn get_mut<'a>(&'a mut self, ch: usize) -> Option<&'a mut [f32]> {
        Some(self.samples[ch].as_mut_slice())
    }

    fn get<'a>(&'a mut self, ch: usize) -> Option<&'a [f32]> {
        Some(self.samples[ch].as_slice())
    }

    fn channels(&self) -> usize {
        self.samples.len()
    }

    fn len(&self) -> usize {
        self.samples[0].len()
    }

    fn data_valid(&self) -> bool {
        true
    }
}


struct DelayLine {
    a: f32,
    buffer: Vec<f32>,
    inptr: usize,
}

impl DelayLine {
    pub fn new(delay_samples: f32) -> Self {
        if delay_samples < 0.0 {
            panic!("Negative delay is not supported");
        }
        let buf_size = f32::floor(delay_samples) as usize + 1;
        let buffer = vec![0.0; buf_size];
        let a = delay_samples - f32::floor(delay_samples);
        Self { a, buffer, inptr: 0 }
    }

    pub fn push(&mut self, sample: f32) -> f32 {
        let nextptr = (self.inptr + 1) % self.buffer.len();
        let pre = self.buffer[self.inptr];
        let post = self.buffer[nextptr];
        let result = pre * self.a + post * (1.0 - self.a);
        self.buffer[self.inptr] = sample;
        self.inptr = nextptr;
        result
    }
}

pub struct WhiteNoiseSource {
    delays: Vec<DelayLine>,
}

impl WhiteNoiseSource {
    /// Create a new multi-channel white noise source with per-channel delay
    /// 
    /// Channel_delays specifies the delay associated with each channel in sample periods.
    /// All delays must be positive.
    /// 
    /// Panics if any delay is negative, or if length of delay slice is 0.
    pub fn new(channel_delays: &[f32]) -> Self {
        if channel_delays.len() == 0 {
            panic!("Cannot create a WhiteNoiseSource with 0 channels");
        }
        let mut result = Self { delays: Vec::new() };

        let mut max_d = -f32::INFINITY;
        for d in channel_delays {
            result.delays.push(DelayLine::new(*d));
            if *d > max_d {
                max_d = *d;
            }
        }

        let n_seed_samples = max_d.ceil() as u32;
        for _ in 0..n_seed_samples {
            // Get random number uniform over (-1, 1)
            let rand_sample:f32 = rand::random::<f32>() * 2.0 - 1.0;

            for ch in 0..result.channels() {
                result.delays[ch].push(rand_sample);
            }
        }
        result
    }

    pub fn next(&mut self, window_size: usize) -> HeapSampleBuffer {
        
        let mut samples = HeapSampleBuffer::new(self.channels(), window_size);

        for i in 0..window_size {
            // Get random number uniform over (-1, 1)
            let rand_sample:f32 = rand::random::<f32>() * 2.0 - 1.0;
            for ch in 0..self.channels() {
                let delay_sample = self.delays[ch].push(rand_sample);
                samples.get_mut(ch).unwrap()[i] = delay_sample;                
            }
        }

        samples
    }

    pub fn channels(&self) -> usize {
        self.delays.len()
    }
}