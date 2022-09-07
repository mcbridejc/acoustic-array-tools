use dasp::{
    sample::Duplex,
    Signal, Frame, Sample
};

pub struct FirFilter<S, const TAPS: usize, const NCHAN: usize> {
    source: S,
    coeffs: [[f32; TAPS]; NCHAN],
    samples: [[f32; TAPS]; NCHAN],
    pos: usize,
}

impl<S, const TAPS: usize, const NCHAN: usize> FirFilter<S, TAPS, NCHAN> {
    pub fn new(source: S, coeffs: [f32; TAPS]) -> Self {
        Self {source: source, coeffs: [coeffs; NCHAN], samples: [[0.0; TAPS]; NCHAN], pos: 0}
    }
}

impl<S, const TAPS: usize, const NCHAN: usize> Signal for FirFilter<S, TAPS, NCHAN> 
where
    S: Signal,
    <<S as Signal>::Frame as Frame>::Sample: Duplex<f32>
{
    type Frame = S::Frame;
    fn next(&mut self) -> Self::Frame {
        let fin = self.source.next();
        
        let fout = Self::Frame::from_fn(|ch| {
            // Copy new sample, replacing oldest sample in buffer
            self.samples[ch][self.pos] = fin.channel(ch).unwrap().to_sample::<f32>();
            let mut total = 0.0;
            for i in 0..TAPS {
                let sample_idx = (self.pos + i) % TAPS;
                total += self.coeffs[ch][i] * self.samples[ch][sample_idx];
            }

            total.to_sample::<<Self::Frame as Frame>::Sample>()
        });
        
        self.pos = (self.pos + 1) % TAPS;
        fout
    }

    fn is_exhausted(&self) -> bool {
        self.source.is_exhausted()
    }
}