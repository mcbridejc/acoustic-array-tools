/** A simple CLI utility to capture audio data from STM32H7 broadcast UDP packets 
into several WAV files (one per channel) */
const NUM_CHANNELS: usize = 1;
const PORT: u16 = 10198;
const MAX_SAMPLES_PER_PACKET: usize = 100;

use hound;
use std::net::{IpAddr, Ipv6Addr, UdpSocket};
use std::str::FromStr;
//use tokio::net::UdpSocket;
use ringbuf::{RingBuffer, Producer, Consumer};
use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use std::fs::File;
use std::io::Write;
use std::io::Read;

const fn make_bit_count_lookup() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut count = 0;
        let mut bit = 0;
        while bit < 8 {
            if (i & (1<<bit)) != 0 {
                count += 1;
            }
            bit += 1;
        }
        table[i] = count;
        i += 1;
    }
    return table;
}

const BIT_COUNT_TABLE: [u8; 256] = make_bit_count_lookup();

struct MovingAverageFilter<const DEC_BYTES: usize> {
    accum: i16,
    buffer: [i16; DEC_BYTES],
    pos: usize,
}

impl<const DEC_BYTES: usize> MovingAverageFilter<DEC_BYTES> {
    pub fn new() -> Self {
        Self { 
            accum: 0,
            buffer: [0i16; DEC_BYTES],
            pos: 0, 
        }
    }

    pub fn push_byte(&mut self, pdm: u8) -> Option<i16> {
        let x = BIT_COUNT_TABLE[pdm as usize] as i16;
        self.accum += x - self.buffer[self.pos];
        self.buffer[self.pos] = x;
        self.pos += 1;
        if self.pos == DEC_BYTES {
            self.pos = 0;
            return Some(self.accum - 4 * DEC_BYTES as i16);
        }
        return None;
    }
}

struct CICFilter<const DECIMATION_BYTES: usize, const STAGES: usize> {
    accum: [i32; STAGES],
    comb: [i32; STAGES],
    pos: usize,
}

impl<const DEC_BYTES: usize, const STAGES: usize> CICFilter<DEC_BYTES, STAGES> {
    pub fn new() -> Self {
        Self { 
            accum: [0i32; STAGES],
            comb: [0i32; STAGES],
            pos: 0 
        }
    }

    pub fn push_byte(&mut self, pdm: u8) -> Option<i16> {
        let mut x = BIT_COUNT_TABLE[pdm as usize] as i32 - 4;
        for i in 0..STAGES {
            self.accum[i] = self.accum[i].overflowing_add(x).0;
            //self.buffer[i][self.pos] = x;
            x = self.accum[i];
        }
        self.pos += 1;
        if self.pos == DEC_BYTES {
            self.pos = 0;
            // Run comb filters at lower sample rate
            // Input is latest output of last integrator stage
            //let mut x = self.accum[STAGES-1];
            
            for i in 0..STAGES {
                let y = x.overflowing_sub(self.comb[i]).0;
                self.comb[i] = x;
                x = y;
            }
            return Some(x as i16)
        }
        return None;
    }

    // pub fn process_multiplexed<const NUM_CHANNELS: usize, const INDEX: usize>(pdm: &[u8]) -> Vec<u16> {
        
    //     let mut samples = Vec::<u16>::with_capacity(pdm.len() / DEC_BYTES / NUM_CHANNELS + 1);

    //     let rdidx = 0;
    //     while rdidx + INDEX*2 < pdm.len() {
            
            
    //     }

    //     return samples;
    // } 

}

struct PacketFileReader {
    file: std::fs::File,
}

impl PacketFileReader {
    pub fn from_file(file: std::fs::File) -> Self {
        Self {
            file
        }
    }
}

impl Iterator for PacketFileReader {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = [0u8; 4];
        let len = match self.file.read_exact(&mut buf) {
            Ok(()) => u32::from_le_bytes(buf) as usize,
            _ => return None,
        };
        let mut data = vec![0; len];
        return match self.file.read_exact(&mut data) {
            Ok(()) => Some(data),
            _ => None,
        }
    }
}

fn main() {
    
    let args: Vec<String> = std::env::args().collect();

    let packet_reader = PacketFileReader::from_file(std::fs::File::open(&args[1]).unwrap());

    let mut last_seq = 0u8;
    

    const PDM_FREQ: u32 = 3072000 / 2;
    const DECIMATION: usize = 1;
    const PCM_FREQ: u32 = PDM_FREQ / DECIMATION as u32 / 8;
        
    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: PCM_FREQ,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("ch1.wav", wav_spec).unwrap();
    let mut ch1_filter = CICFilter::<DECIMATION, 4>::new();
    //let mut ch1_filter = MovingAverageFilter::<DECIMATION>::new();

    for packet in packet_reader {
        let seq = packet[packet.len() - 2];
        if seq != last_seq.overflowing_add(1).0 {
            println!("Missed sequence {} -> {}", last_seq, seq);
        }
        last_seq = seq;

        let flags = packet[packet.len() - 1];
        if flags != 0 {
            println!("OVF");
        }
        let mut idx = 0;
        loop {
            // Last two bytes are metadata footers
            if idx >= packet.len() - 2 {
                break;
            }
            
            if let Some(sample) = ch1_filter.push_byte(packet[idx]) {
                writer.write_sample(sample.overflowing_mul(1).0).unwrap();
            }
            idx += NUM_CHANNELS;
        }
    }

    writer.finalize().unwrap();

}

