#[allow(dead_code)]

use crate::device;

/// Configure the DMAMUX1, which routes peripheral requests for DMA1/DMA2
/// Channel 0-7 go to DMA1, channel 8-15 to DMA2.
/// See ref manual for complete list of peripheral request IDs.
pub fn set_dma_request_mux(channel: usize, request: u8) {
    assert!(channel < 16);
    let dmamux = unsafe { device::Peripherals::steal().DMAMUX1 }; 
    dmamux.ccr[channel].modify(|_, w| { 
        unsafe { w.dmareq_id().bits(request & 0x7f).ege().enabled() }
    });
}

pub enum PriorityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    VeryHigh = 3,
}

pub enum DataSize {
    Byte = 0,
    HalfWord = 1,
    Word = 2,
}

#[repr(u8)]
#[derive(PartialEq)]
pub enum TargetBuffer {
    Buffer0 = 0,
    Buffer1 = 1,
}

pub struct DmaStream<const IDX: usize> {
    reg: &'static device::dma1::RegisterBlock,
}

impl<const IDX: usize> DmaStream<IDX> {
    pub fn new(reg: &'static device::dma1::RegisterBlock) -> Self {
        Self { reg }
    }

    /** Set Priority Level. This can only be written when the stream is disabled. */
    pub fn set_priority_level(&self, prio: PriorityLevel) {
        self.reg.st[IDX].cr.modify(|_, w| {
            w.pl().bits(prio as u8)
        });
    }

    /** Set peripheral transfer size. This can only be written when the stream is disabled */
    pub fn set_psize(&self, size: DataSize) {
        self.reg.st[IDX].cr.modify(|_, w| {
            unsafe { w.psize().bits(size as u8) }
        });
    }

    pub fn get_current_target(&self) -> TargetBuffer {
        if self.reg.st[IDX].cr.read().ct().is_memory0() {
            TargetBuffer::Buffer0
        } else {
            TargetBuffer::Buffer1
        }
    }

    pub fn start_p2m_transfer(&self, periph_addr: usize, buf: *mut [u8],  second_buf: Option<*mut [u8]>, length: usize) {
        self.reg.st[IDX].cr.modify(|_, w| {
            let w = w.ct().memory0()
            .minc().incremented()
            .pinc().fixed()
            .pfctrl().dma()
            .tcie().enabled();
            if second_buf.is_some() {
                w.dbm().enabled()
            } else {
                w.dbm().disabled()
            }
        });
        
        self.reg.st[IDX].m0ar.write(|w| {
            unsafe { w.m0a().bits(buf as *mut () as u32) }
        });

        self.reg.st[IDX].par.write(|w| {
            unsafe { w.pa().bits(periph_addr as u32) }
        });

        if second_buf.is_some() {
            self.reg.st[IDX].m1ar.write(|w| {
                unsafe { w.m1a().bits(second_buf.unwrap() as *mut () as u32) }
            });
        }

        let cr = self.reg.st[IDX].cr.read();
        let ndt = if cr.psize().is_bits16() {
            length / 2
        } else if cr.psize().is_bits32() {
            length / 4
        } else {
            length
        };
        self.reg.st[IDX].ndtr.write(|w| {
            w.ndt().bits(ndt as u16)
        });

        self.reg.st[IDX].cr.modify(|_, w| {
            w.en().enabled()
        });
    }

    pub fn load_memory0(&self, buf: *mut [u8]) {
        self.reg.st[IDX].m0ar.write(|w| {
            unsafe { w.m0a().bits(buf as *mut () as u32) }
        })
    }

    pub fn load_memory1(&self, buf: *mut [u8]) {
        self.reg.st[IDX].m1ar.write(|w| {
            unsafe { w.m1a().bits(buf as *mut () as u32) }
        })
    }

    pub fn enable_interrupt(&self) {
        // TODO: SUPPORT OTHER INTERRUPTS FLAGS :)
        // Sorry world for lazy
        self.reg.st[IDX].cr.modify(|_, w| {
            w.tcie().enabled()
        });
    }

    pub fn clear_interrupts(&self) {
        // Clear all pending flags
        if IDX > 3 {
            self.reg.hifcr.write(|w| {
                unsafe { w.bits(0b111101 << (6 * (IDX - 4))) }
            });
        } else {
            self.reg.lifcr.write(|w| {
                unsafe { w.bits(0b111101 << (6 * (IDX))) }
            });
        }
    }
}