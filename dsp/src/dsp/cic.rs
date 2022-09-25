
pub fn demux_pdm_all_channels<F, const N_CHAN: usize>(pdm: &[u8], mut output: F)
where F: FnMut([f32; N_CHAN])
{
    assert!(pdm.len() % N_CHAN == 0);

    let mut offset = 0usize;

    while offset < pdm.len() - N_CHAN {
        let pdm_frame = unsafe { pdm.get_unchecked(offset..offset+N_CHAN) };
        //let word = unsafe {pdm.get_unchecked(offset+ch)};
        for bit in 0..8 {
            let mut frame = [-1.0; N_CHAN];
            for ch in 0..N_CHAN {
                if ( pdm_frame[ch] & (1<<bit)) != 0 {
                    frame[ch] = 1.0;
                }
            }
            output(frame);
        }
        offset += N_CHAN;
    }
}

pub fn demux_pdm_one_channel<F, const N_CHAN: usize>(pdm: &[u8], ch: usize, mut output: F)
where F: FnMut(f32)
{
    assert!(pdm.len() % N_CHAN == 0);
    assert!(ch < N_CHAN);

    let mut offset = 0usize;

    while offset < pdm.len() {
        let mut frame = -1.0;
        let word = unsafe {pdm.get_unchecked(offset+ch)};
        for bit in 0..8 {
            if (word & (1<<bit)) != 0 {
                frame = 1.0;
            }
            output(frame);
        }
        offset += N_CHAN;
    }
}

#[derive(Copy, Clone)]
pub struct CicFilter<
    const DECIMATION: usize,
    const STAGES: usize,
    >
{
    integrator: [i32; STAGES],
    comb: [i32; STAGES],
    pos: usize,
}

impl<
    const DECIMATION: usize,
    const STAGES: usize,
> CicFilter<DECIMATION, STAGES> {
    pub fn new() -> Self {
        Self {
            integrator: [0; STAGES],
            comb: [0; STAGES],
            pos: 0
        }
    }

    fn integrate(&mut self, value: i32) -> i32 {
        let mut x = value;
        for stage in 0..STAGES {
            self.integrator[stage] = self.integrator[stage].overflowing_add(x).0;
            x = self.integrator[stage];
        }
        x
    }

    fn comb(&mut self) -> i32 {
        // Last integrator stage is always input
        let mut x = self.integrator[STAGES - 1];
        for stage in 0..STAGES {
            let y = x.overflowing_sub(self.comb[stage]).0;
            self.comb[stage] = x;
            x = y;
        }
        x
    }

    /// Process samples from a multi-channel PDM byte buffer, which contains NCHAN channels.
    /// Only the channel given by `channel` argument is read; other channels are ignored.
    pub fn process_pdm_buffer<F, const NCHAN: usize>(&mut self, channel: usize, pdm: &[u8], mut output: F)
    where F: FnMut(i32)
    {
        assert!(pdm.len() % NCHAN == 0);

        let mut offset = 0usize;

        while offset < pdm.len() - NCHAN {
            let pdm_frame = unsafe { pdm.get_unchecked(offset..offset+NCHAN) };

            for bit in 0..8 {
                if (pdm_frame[channel] & (1<<bit)) != 0 {
                    self.integrate(1)
                } else {
                    self.integrate(-1)
                };
                self.pos += 1;
                if self.pos == DECIMATION {
                    self.pos = 0;
                    output(self.comb());
                }
            }
            offset += NCHAN;
        }
    }

    pub fn push_sample<F>(&mut self, sample: i32, mut output: F)
    where F: FnMut(i32)
    {
        self.integrate(sample);

        self.pos += 1;
        if self.pos == DECIMATION {
            self.pos = 0;
            output(self.comb());
        }
    }
}


