use std::sync::{ 
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::net::{UdpSocket, Ipv4Addr};
use std::str::FromStr;
use crossbeam_channel::Sender;
use std::time::Duration;

const PORT: u16 = 10200;
const MAX_PACKET_SIZE: usize = 512;

pub fn udp_rx_task(break_signal: Arc<AtomicBool>, packet_tx: Sender<Vec<u8>>) {
    let broadcast_ip = Ipv4Addr::from_str("255.255.255.255").unwrap();
    let ip = Ipv4Addr::from_str("0.0.0.0").unwrap();
    let socket = UdpSocket::bind((ip, PORT)).unwrap();
    socket.set_read_timeout(Some(Duration::from_millis(50))).unwrap();

    loop {
        if break_signal.load(Ordering::Relaxed) {
            return;
        }
    
        let mut buf = [0u8; MAX_PACKET_SIZE];
        if let Ok((amt, _src)) = socket.recv_from(&mut buf) {
            packet_tx.send(Vec::from(&buf[0..amt])).unwrap();
        }
    }
}