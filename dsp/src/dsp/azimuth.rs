use num_complex::Complex;

/// Routines for determining azimuth of a source from a circular array of power measurements
/// 

/// Compute a power "moment" from a series of power measurements at different azimuth
/// 
/// It is assumed that the first measurement in az_powers is at 0 degrees azimuth, and that the
/// points are then spread equally over the full circle, in order of increasing azimuth, and evenly
/// spaced.
pub fn weighted_azimuth(az_powers: &[f32]) -> Complex<f32> {
    let mut result = Complex{re: 0.0, im: 0.0};
    for i in 0..az_powers.len() {
        let theta = core::f32::consts::PI * 2.0 * i as f32 / az_powers.len() as f32;
        result += Complex::from_polar(az_powers[i], theta);
    }
    result
}

pub struct AzFilter {
    sample_count: usize,
    moment_count: usize,
    moment: Complex<f32>,
    dynamic_threshold: f32,
    rms_accum: f32,
    rms_thresh: f32,
    rms_decay: f32
}

impl AzFilter {

    /// Create a new filter
    /// 
    /// The filter takes as input the series of moments calculated from a series of radial focal
    /// points. Moment is a complex number calculated as the sum over all power SUM over i: (P[i] *
    /// cos(theta[i]) + j * P[i] * sin(theta[i])))
    /// 
    /// Where P[i] is the ith measurement, and theta[i] is the azimuth angle of the ith focal point.
    /// (j is sqrt(-1) here)
    /// 
    /// The goal is to determine when a new "event" is detected and at what direction, so that a
    /// pointing device of some kind can be pointed towards the source of the sound. This is a crude
    /// filter that seems to do a reasonable job of fixing some of the problems that came up with an
    /// even more naive approach of return the peak angle of any loud frame. One of the problems is
    /// that loud sounds are often followed by echos in a following frame which are not generally in
    /// the same direction as the source. These echos can be loud enough to surpass a reasonable RMS
    /// threshold, so that often one would see a brief solution in the correct direction, followed
    /// immediately by an erroneous one. Here, the threshold is increased temporarily after a
    /// solution so that only the loudest sound heard recently will be considered. 
    /// 
    /// - rms_thresh: The activity threshold; frames with RMS lower than this are ignored 
    /// 
    /// - rms_decay: Rate of decay of the dynamically increased RMS threshold (dB / frame)
    /// 
    /// I have found rms_thresh = -55.0, and rms_decay = 0.25 to be reasonable values, but these
    /// values and the general filter approach probably deserve further consideration.
    pub const fn new(rms_thresh: f32, rms_decay: f32) -> Self {
        Self {
            sample_count: 0,
            moment_count: 0,
            moment: Complex { re: 0.0, im: 0.0 },
            dynamic_threshold: rms_thresh,
            rms_accum: 0.0,
            rms_thresh,
            rms_decay,
        }
    }

    pub const fn default() -> Self {
        Self {
            sample_count: 0,
            moment_count: 0,
            moment: Complex { re: 0.0, im: 0.0 },
            dynamic_threshold: -55.0,
            rms_accum: 0.0,
            rms_thresh: -55.0,
            rms_decay: 0.25,
        }
    }

    pub fn push(&mut self, moment: Option<Complex<f32>>, rms: f32) -> Option<f32> {
        if self.dynamic_threshold > self.rms_thresh {
            self.dynamic_threshold -= self.rms_decay;
        }
        if self.dynamic_threshold < self.rms_thresh {
            self.dynamic_threshold = self.rms_thresh;
        }

        if rms >= self.dynamic_threshold {
            // If this is the first sample in an estimate, don't start collecting without a moment
            // Once we've started estimating, we will accept empty frames -- frames which had an
            // above-threshold RMS but were not processed -- for the purpose of counting or
            // finishing an estimate even though we cannot acummulate the moment.
            if moment.is_some() || self.sample_count > 0 {
                self.sample_count += 1;
                self.rms_accum += rms;
            }

            if moment.is_some() {
                self.moment += moment.unwrap();
                self.moment_count += 1;
            }
            if self.sample_count > 25 {
                return self.complete();
            }
        } else {
            if self.moment_count > 0 {
                return self.complete();
            }
        }

        None
    }

    pub fn complete(&mut self) -> Option<f32> {
        let (mag, angle) = self.moment.to_polar();
        let result = if mag > 20.0 {
            Some(angle)
        } else {
            None
        };

        self.dynamic_threshold = self.rms_accum / self.sample_count as f32;
        self.sample_count = 0;
        self.moment_count = 0;
        self.moment = Complex{ re: 0.0, im: 0.0 };
        self.rms_accum = 0.0;

        result
    }
}