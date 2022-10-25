use panic_itm as _;
use lazy_static::lazy_static;
use log::LevelFilter;

pub use cortex_m_log::log::Logger;

use cortex_m_log::{
    destination::Itm as ItmDest,
    destination::Dummy as DummyDest,
    printer::itm::InterruptSync,
    modes::InterruptFree,
    printer::itm::ItmSync,
    printer::dummy::Dummy,
};

lazy_static! {
    static ref LOGGER: Logger<ItmSync<InterruptFree>> = Logger {
        level: LevelFilter::Info,
        inner: unsafe {
            InterruptSync::new(
                ItmDest::new(cortex_m::Peripherals::steal().ITM)
            )
        },
    };
}

// lazy_static! {
//     static ref LOGGER: Logger<Dummy> = Logger {
//         level: LevelFilter::Info,
//         inner: Dummy::new(),
        
//     };
// }


pub fn init() {
    cortex_m_log::log::init(&LOGGER).unwrap();
}