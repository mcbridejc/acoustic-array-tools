
#[allow(dead_code)]

// A driver for reading PDM microphones using STM32H7 SAI cores
use crate::device as device;

pub enum FifoThreshold {
    Empty = 0,
    Quarter = 1,
    Half = 2,
    ThreeQuarter = 3,
    Full = 4
}
pub enum Interrupt {
    OvrUdr = (1<<0),
    MuteDet = (1<<1),
    WCkCfg = (1<<2),
    FReq = (1<<3),
    CNRdy = (1<<4),
    AFSDet = (1<<5),
    LFSDet = (1<<6),
}


pub struct PdmInput {
    sai1: &'static device::sai4::RegisterBlock,
}

impl PdmInput {
    pub fn new(
        sai1: &'static device::sai4::RegisterBlock,
    ) -> Self {
        Self {
            sai1: sai1
        }
    }

    pub fn init(&mut self) {
        // FOR 6 Channels
        // Read in 3 16-bit slots
        self.sai1.cha.cr1.modify(|_, w| {
            unsafe { w.mode().master_rx().nomck().set_bit().ds().bits(0b100).mckdiv().bits(0).lsbfirst().set_bit() }
        });
        self.sai1.cha.frcr.modify(|_, w| {
            unsafe { w.fspol().set_bit().frl().bits(47) }
        });
        self.sai1.cha.slotr.modify(|_, w| {
            unsafe { w.nbslot().bits(2).sloten().bits(0b111) }
        });
        self.sai1.pdmcr.modify(|_, w| {
            unsafe{ w.micnbr().bits(0b10).cken1().set_bit() }
        });
        self.sai1.pdmcr.modify(|_, w| {
            w.pdmen().set_bit()
        });
        self.sai1.pdmdly.modify(|_, w| {
            unsafe {
                w.dlym1l().bits(2)
                .dlym1r().bits(2)
                .dlym2l().bits(2)
                .dlym2r().bits(2)
                .dlym3l().bits(2)
                .dlym3r().bits(2)
                .dlym4l().bits(2)
                .dlym4r().bits(2)
            }
        });

        self.sai1.cha.cr1.modify(|_, w| {
            w.saien().set_bit() 
        }); 

    
        // FOR A SINGLE CHANNEL
        // self.sai1.cha.cr1.modify(|_, w| {
        //     unsafe { w.mode().master_rx().nomck().set_bit().ds().bit8().mckdiv().bits(3) }
        // });
        // self.sai1.cha.frcr.modify(|_, w| {
        //     unsafe { w.fspol().set_bit().frl().bits(15) }
        // });
        // self.sai1.cha.slotr.modify(|_, w| {
        //     unsafe { w.nbslot().bits(1).sloten().bits(0b10) }
        // });
        // self.sai1.pdmcr.modify(|_, w| {
        //     unsafe{ w.micnbr().bits(0b00).cken1().set_bit() }
        // });
        // self.sai1.pdmcr.modify(|_, w| {
        //     w.pdmen().set_bit()
        // });

        // self.sai1.cha.cr1.modify(|_, w| {
        //     w.saien().set_bit() 
        // }); 
    }

    pub fn data_available(&mut self) -> bool {
        let flvl = self.sai1.cha.sr.read().flvl().bits();
        return flvl != 0;
    }

    pub fn overflow_flag(&mut self) -> bool {
        let flag = self.sai1.cha.sr.read().ovrudr().bit_is_set();
        if flag {
            self.sai1.cha.clrfr.write(|w| {
                w.covrudr().set_bit()
            });
        }
        flag
    }

    pub fn read(&mut self) -> u32 {
        return self.sai1.cha.dr.read().bits();
    }

    pub fn set_fifo_threshold(&mut self, thr: FifoThreshold) {
        self.sai1.cha.cr2.modify(|_, w| {
            unsafe { w.fth().bits(thr as u8)}
        });
    }

    pub fn enable_interrupts(&mut self, intmask: Interrupt) {
        self.sai1.cha.im.modify(|r, w| {
            unsafe { w.bits(r.bits() | (intmask as u32)) }
        });
    }

    pub fn disable_interrupts(&mut self, intmask: Interrupt) {
        self.sai1.cha.im.modify(|r, w| {
            unsafe { w.bits(r.bits() & !(intmask as u32)) }
        });
    }

    pub fn enable_dma(&mut self) {
        self.sai1.cha.cr1.modify(|_, w| {
            w.dmaen().enabled()
        });
    }
}

use crate::hal::dma::traits::{TargetAddress};
// Add trait to make it addressable by the HAL dma driver
unsafe impl TargetAddress<crate::hal::dma::PeripheralToMemory> for PdmInput {

    type MemSize = u16;

    const REQUEST_LINE: Option<u8> = Some(87);
    fn address(&self) -> usize {
        &self.sai1.cha.dr as *const _ as usize

    }
}
