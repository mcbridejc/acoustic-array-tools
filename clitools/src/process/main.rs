/** A simple CLI utility to capture audio data from STM32H7 broadcast UDP packets 
into several WAV files (one per channel) */
const NUM_CHANNELS: usize = 6;

use hound;
use std::fs::File;
use std::io::Read;

use dasp::{Frame, Signal};
use dsp::{
    dasp::{BiquadFilter, CicDecimator, PdmSource},
    fir::{FirFilter}
};


use dsp::dasp::StreamingIterator;

struct PacketFileReader<'a, const MAX_PACKET_SIZE: usize = 512> {
    file: File,
    buffer: [u8; MAX_PACKET_SIZE],
    size: usize,
    empty: bool,
    _lifetime: core::marker::PhantomData<&'a ()>,
}

impl<'a, const MAX_PACKET_SIZE: usize> PacketFileReader<'a, MAX_PACKET_SIZE> {
    pub fn from_file(file: File) -> Self {
        Self {
            file: file,
            buffer: [0u8; MAX_PACKET_SIZE],
            size: 0,
            empty: false,
            _lifetime: core::marker::PhantomData,
        }
    }
}

impl<'a, const MAX_PACKET_SIZE: usize> StreamingIterator for PacketFileReader<'a, MAX_PACKET_SIZE> 
{
    type Item = u8;

    fn get<'b>(&'b self) -> Option<&'b [Self::Item]> {
        if self.empty {
            None
        } else {
            Some(&(*self).buffer[0..self.size])
        }
    }

    fn next<'b>(&'b mut self) -> Option<&'b [Self::Item]> {
        if self.empty {
            return None;
        } else {
            return Some(&self.buffer[0..self.size]);
        }
    }

    fn advance<'b>(&'b mut self) {
        let mut buf = [0u8; 4];
        let mut len = match self.file.read_exact(&mut buf) {
            Ok(()) => u32::from_le_bytes(buf) as usize,
            _ => {
                self.empty = true;
                return;
            }
        };
        if len > MAX_PACKET_SIZE {
            len = MAX_PACKET_SIZE;
        }
        
        match self.file.read_exact(&mut self.buffer[0..len]) {
            Ok(()) => {
                self.size = len;
                self.empty = false;
            },
            _ => self.empty = true
        }
    }
}

fn main() {
    
    use biquad::*;

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Must provide two args: process <infile> <outprefix>");
        return;
    }

    let mut packet_reader = PacketFileReader::<512>::from_file(std::fs::File::open(&args[1]).unwrap());

    let outputname = &args[2];

    const PDM_FREQ: u32 = 3072000;
    const DECIMATION: usize = 128;
    const PCM_FREQ: u32 = PDM_FREQ / DECIMATION as u32;
        
    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: PCM_FREQ,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Converts sequence of packets into sequence of N channel frames
    let frame_iter = dsp::dasp::FrameIterator::<NUM_CHANNELS, _>::new(&mut packet_reader);

    let hp_coeffs = Coefficients::<f32>::from_params(
        biquad::Type::HighPass,
        (PDM_FREQ / DECIMATION as u32).hz(), // sample freq
        50u32.hz(), // cutoff freq
        biquad::Q_BUTTERWORTH_F32
    ).unwrap();

    const LOWPASS_COEFFS: [f32; 51] = [
        -0.00095132,  0.00373081,  0.00501343,  0.00611545,  0.0054415,
        0.00240435, -0.00246896, -0.00753223, -0.01047923, -0.00931311,
        -0.00337049,  0.00592721,  0.01516248,  0.0199621 ,  0.01673867,
        0.00454584, -0.01375828, -0.03166564, -0.0407084 , -0.0332403 ,
        -0.00540342,  0.04081833,  0.09735496,  0.15189371,  0.19133069,
        0.20571242,  0.19133069,  0.15189371,  0.09735496,  0.04081833,
        -0.00540342, -0.0332403 , -0.0407084 , -0.03166564, -0.01375828,
        0.00454584,  0.01673867,  0.0199621 ,  0.01516248,  0.00592721,
        -0.00337049, -0.00931311, -0.01047923, -0.00753223, -0.00246896,
        0.00240435,  0.0054415 ,  0.00611545,  0.00501343,  0.00373081,
        -0.00095132
    ];


    let pdm = PdmSource::new(dasp::signal::from_iter(frame_iter));
    let dec1 = CicDecimator::<_, 8, 4>::new(pdm);
    let dec2 = CicDecimator::<_, 4, 3>::new(dec1);
    let hp_filter = BiquadFilter::<_, NUM_CHANNELS>::new(dec2, hp_coeffs);
    let fir_filter = FirFilter::<_, 51, NUM_CHANNELS>::new(hp_filter, LOWPASS_COEFFS); 
    let mut dec3 = dsp::dasp::CicDecimator::<_, 4, 2>::new(fir_filter);
    let mut writers = Vec::new();
    

    for i in 0..NUM_CHANNELS {
        let filename = format!("{}_{}.wav", outputname, i);
        writers.push(hound::WavWriter::create(filename, wav_spec).unwrap());
    }
    
    while !dec3.is_exhausted() {
        let frame: [i32; NUM_CHANNELS] = dec3.next();
        for ch in 0..NUM_CHANNELS {
            writers[ch].write_sample((*frame.channel(ch).unwrap() / 4) as i16).unwrap(); 
        }
    }

}

