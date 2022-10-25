

// TODO: Storing coeffs by value here is wasting memory when multiple channels are processed with
// the same filter We can store one reference; it's probably static, and even if not we should be
// able to ensure the coeffs outlive the filter

#[derive(Clone, Copy)]
pub struct FloatFir<const TAPS: usize> {
    coeffs: [f32; TAPS],
    samples: [f32; TAPS],
    pos: usize
}

impl<const TAPS: usize> FloatFir<TAPS> {
    pub const fn new(coeffs: [f32; TAPS]) -> Self {
        Self {
            coeffs: coeffs,
            samples: [0.0; TAPS],
            pos: 0
        }
    }

    pub fn process_sample(&mut self, sample: f32) -> f32 {
        self.samples[self.pos] = sample;
        self.pos = (self.pos + 1) % TAPS;
        self.output()
    }

    pub fn output(&self) -> f32 {
        let mut y = 0.0f32;

        for i in 0..TAPS {
            let sample_idx = (self.pos + i) % TAPS;
            y += self.coeffs[i] * self.samples[sample_idx];
        }
        y
    }
}