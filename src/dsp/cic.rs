

impl<const DECIMATION: usize, const STAGES: usize> CICFilter<DECIMATION, STAGES> {
    pub fn new() -> Self {
        Self { 
            accum: [0i16; STAGES],
            comb: [0i16; STAGES],
            pos: 0 
        }
    }

    pub fn push_byte(&mut self, pdm: u8, mut rx: impl FnMut(i16) ) {
        for i in 0..8 {
            let mut x = ((pdm & (1<<i)) >> i) as i16;
            for i in 0..STAGES {
                self.accum[i] = self.accum[i].overflowing_add(x).0;
                x = self.accum[i];
            }
            self.pos += 1;
            if self.pos == DECIMATION {
                self.pos = 0;
                // Run comb filters at lower sample rate
                // Input is latest output of last integrator stage
                for i in 0..STAGES {
                    let y = x.overflowing_sub(self.comb[i]).0;
                    self.comb[i] = x;
                    x = y;
                }
                rx(x);
            }
        }
    }
}
