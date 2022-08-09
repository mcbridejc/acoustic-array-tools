/** A simple CLI utility to capture audio data from STM32H7 broadcast UDP packets 
into several WAV files (one per channel) */
const PORT: u16 = 10198;
const MAX_PACKET_SIZE: usize = 500;

use std::net::{IpAddr, Ipv6Addr, UdpSocket};
use std::str::FromStr;
use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use std::fs::File;
use std::io::Write;

fn main() -> io::Result<()> {

    let ctrlc_signal = Arc::<AtomicBool>::new(AtomicBool::new(false));
    let ctrlc_signal2 = ctrlc_signal.clone();
    let ctrlc_signal3 = ctrlc_signal.clone();
    ctrlc::set_handler(move || {
        println!("Finalizing wav files...");
        ctrlc_signal2.store(true, std::sync::atomic::Ordering::Relaxed)
    }).unwrap();

    
    let (pkt_tx, pkt_rx): (
        mpsc::Sender::<(usize, [u8; MAX_PACKET_SIZE])>,
        mpsc::Receiver::<(usize, [u8; MAX_PACKET_SIZE])>) = mpsc::channel();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Must provide an output file");
        return Ok(());
    }


    let mut outfile = File::create(&args[1]).expect("Failed opening file");
    let write_thread = thread::spawn(move || {
        loop {
            let break_sig = ctrlc_signal3.load(std::sync::atomic::Ordering::Relaxed);
            if break_sig {
                break;
            }

            if let Ok((amt, buf)) = pkt_rx.recv_timeout(Duration::from_millis(50)) {
                outfile.write_all(&(amt as u32).to_le_bytes()).unwrap();
                outfile.write_all(&buf[0..amt]).unwrap();
            }
        }
        outfile.flush().unwrap();
    });

    //let ip = IpAddr::V6(Ipv6Addr::from_str("fe80::e73:8433:2f2d:76c").unwrap());
    let ip = IpAddr::V6(Ipv6Addr::from_str("0::").unwrap());
    let socket = UdpSocket::bind((ip, PORT)).unwrap();
    let multicast_ip = Ipv6Addr::from_str("ff02::1").unwrap();
    socket.join_multicast_v6(&multicast_ip, 0).unwrap();

    loop {
        let break_sig = ctrlc_signal.load(std::sync::atomic::Ordering::Relaxed);
        if break_sig {
            break;
        }
        let mut buf = [0u8; MAX_PACKET_SIZE];
        let (amt, _src) = socket.recv_from(&mut buf).unwrap();
        
        pkt_tx.send((amt, buf)).unwrap();
        
    }

    write_thread.join().unwrap();


    return Ok(())

}

