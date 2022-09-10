
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
    let mut bit = 0usize;

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

pub struct CicFilter<
    const DECIMATION: usize,
    const STAGES: usize,
    const N_CHAN: usize,
    >
{
    integrator: [[i32; STAGES]; N_CHAN],
    comb: [[i32; STAGES]; N_CHAN],
    pos: usize,
}

impl<
    const DECIMATION: usize,
    const STAGES: usize,
    const N_CHAN: usize,
> CicFilter<DECIMATION, STAGES, N_CHAN> {
    pub fn new() -> Self {
        Self { 
            integrator: [[0; STAGES]; N_CHAN],
            comb: [[0; STAGES]; N_CHAN],
            pos: 0
        }
    }

    fn integrate(&mut self, ch: usize, value: i32) -> i32 {
        let mut x = value;
        for stage in 0..STAGES {
            self.integrator[ch][stage] = self.integrator[ch][stage].overflowing_add(x).0;
            x = self.integrator[ch][stage];
        }
        x
    }
    
    fn comb(&mut self, ch: usize) -> i32 {
        // Last integrator stage is always input
        let mut x = self.integrator[ch][STAGES - 1];
        for stage in 0..STAGES { 
            let y = x.overflowing_sub(self.comb[ch][stage]).0;
            self.comb[ch][stage] = x;
            x = y;
        }
        x
    }

    pub fn process_pdm_buffer<F>(&mut self, pdm: &[u8], mut output: F) 
    where F: FnMut([i32; N_CHAN])
    {
        assert!(pdm.len() % N_CHAN == 0);

        let mut offset = 0usize;

        while offset < pdm.len() - N_CHAN {
            let pdm_frame = unsafe { pdm.get_unchecked(offset..offset+N_CHAN) };
            
            for bit in 0..8 {
                for ch in 0..N_CHAN {
                    let x = if (pdm_frame[ch] & (1<<bit)) != 0 {
                        self.integrate(ch, 1)
                    } else {
                        self.integrate(ch, -1)
                    };
                }
                self.pos += 1;
                if self.pos == DECIMATION {
                    self.pos = 0;
                    let mut frame = [0; N_CHAN];
                    for ch in 0..N_CHAN {
                        frame[ch] = self.comb(ch);
                    }
                    output(frame);
                }
            }
            offset += N_CHAN;
        }
    }

    pub fn push_sample<F>(&mut self, sample: [i32; N_CHAN], mut output: F)
    where F: FnMut([i32; N_CHAN])
    {
        for i in 0..N_CHAN {
            self.integrate(i, unsafe {*sample.get_unchecked(i)} );
        }
        self.pos += 1;
        if self.pos == DECIMATION {
            self.pos = 0;
            let mut frame = [0; N_CHAN];
            for ch in 0..N_CHAN {
                frame[ch] = self.comb(ch);
            }
            output(frame);
        }
    }

    pub fn process_all_channels<F>(&mut self, pdm: &[u8], output: F)
    where F: FnMut([f32; N_CHAN])
    {
        

    }    
}


