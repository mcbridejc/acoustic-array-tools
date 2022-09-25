
// use dasp::{Sample, Signal, Frame, sample::Duplex};


// pub struct PdmSource<S, F> 
// where 
//     S: Signal,
//     <S as Signal>::Frame: Frame<Sample=u8>,
// {
//     source: S,
//     last_frame: S::Frame,
//     bit: usize,
//     _marker: core::marker::PhantomData<F>,
// }

// impl<S, F> PdmSource<S, F>
// where
//     S: Signal,
//     <S as Signal>::Frame: Frame<Sample=u8>,
//     F: Frame
// {
//     pub fn new(source: S) -> Self{
//         Self { source: source, last_frame: S::Frame::EQUILIBRIUM, bit: 0, _marker: core::marker::PhantomData }
//     }
// }

// impl<S, F> Signal for PdmSource<S, F> 
// where
//     S: Signal,
//     <S as Signal>::Frame: Frame<Sample=u8>,
//     F: Frame<Sample = i32>,
// {
//     type Frame = F;

//     fn next(&mut self) -> Self::Frame {
//         if self.bit == 0 {
//             if self.source.is_exhausted() {
//                 return Self::Frame::EQUILIBRIUM;
//             } else {
//                 self.last_frame = self.source.next();
//             }
//         }

//         let out = Self::Frame::from_fn(|i| {
//             let pdm_byte = unsafe { self.last_frame.channel_unchecked(i) };
//             // Get bit as 0 or 1
//             let y = ((*pdm_byte & (1<<(self.bit))) >> (self.bit)) as i32;
//             // Transform to -1/+1 instead of 0,1 to remove DC bias
//             y * 2 - 1
//             //((((*pdm_byte as i16) & (1<<self.bit)) >> (self.bit-1)) - 1) as i16
//         });
//         self.bit = (self.bit + 1) % 8;
//         out
//     }

//     fn is_exhausted(&self) -> bool {
//         self.source.is_exhausted()
//     }
// }


// struct CicFrameDecimator<F, const STAGES: usize>
//     where F: Frame,
//     F::Sample: Duplex<i32>,
// {
//     integral: [F; STAGES],
//     comb: [F; STAGES],
// }

// impl<F, const STAGES: usize> CicFrameDecimator<F, STAGES> 
//     where F: Frame,
//     F::Sample: Duplex<i32>,
// {
//     fn next_source_frame(&mut self, x: F) {
//         let mut x = x;
//         for s in 0..STAGES {
//             self.integral[s] = self.integral[s].zip_map(x, |a, b| {
//                 let a_i = a.to_sample::<i32>();
//                 let b_i = b.to_sample::<i32>();
//                 (a_i.overflowing_add(b_i).0).to_sample::<F::Sample>()
//             });
            
//             x = self.integral[s];
//         }
//     }

//     fn interpolate(&mut self) -> F {
//         let mut x = self.integral[self.integral.len() - 1];
//         for s in 0..STAGES {
//             let y = self.comb[s].zip_map(x, |a, b| {
//                 let a_i = a.to_sample::<i32>();
//                 let b_i = b.to_sample::<i32>();
//                 (b_i.overflowing_sub(a_i)).0.to_sample::<F::Sample>()
//             });
//             self.comb[s] = x;
//             x = y;
//         }
//         x
//     }
// }

// // Like a dasp Converter, but without double math. It only supports integer decimation ratios
// pub struct CicDecimator <S, const D: usize, const ORDER: usize>
// where 
//     S: Signal,
//     <<S as Signal>::Frame as Frame>::Sample: Duplex<i32>,
// {
//     source: S,
//     decimator: CicFrameDecimator<S::Frame, ORDER>,
// }

// impl <S, const D: usize, const ORDER: usize> CicDecimator<S, D, ORDER>
// where 
//     S: Signal,
//     <<S as Signal>::Frame as Frame>::Sample: Duplex<i32>,
// {
//     pub fn new(source: S) -> Self {
//         Self {
//             source: source,
//             decimator: CicFrameDecimator { integral: [S::Frame::EQUILIBRIUM; ORDER], comb: [S::Frame::EQUILIBRIUM; ORDER] }
//         }
//     }
// }

// impl <S, const D: usize, const ORDER: usize> Signal for CicDecimator<S, D, ORDER>
// where
//     S: Signal,
//     <<S as Signal>::Frame as Frame>::Sample: Duplex<i32>,
// {
//     type Frame = S::Frame;
    
//     fn next(&mut self) -> Self::Frame {
//         for _ in 0..D {
//             let frame = self.source.next();
//             self.decimator.next_source_frame(frame);
//         }
//         self.decimator.interpolate()
//     }

//     fn is_exhausted(&self) -> bool {
//         self.source.is_exhausted()
//     }
// }

// pub trait StreamingIterator {
//     type Item;

//     fn advance<'b>(&'b mut self);

//     fn next<'b>(&'b mut self) -> Option<&'b [Self::Item]>;

//     fn get<'b>(&'b self) -> Option<&'b [Self::Item]>;
// }

// pub struct FrameIterator<'a, const NCHAN: usize, I>
// where I: StreamingIterator<Item=u8>
// {
//     pos: usize,
//     packet_iter: &'a mut I, // &'a mut dyn Iterator<Item = &'a [u8]>,
//     _lifetime: core::marker::PhantomData<&'a ()>,
// }

// impl<'a, const NCHAN: usize, I> FrameIterator<'a, NCHAN, I >
// where
//     I: StreamingIterator<Item = u8>
// {
//     pub fn new(packet_iter: &'a mut I) -> Self {
//         Self { 
//             pos: 0,
//             packet_iter: packet_iter,
//             _lifetime: core::marker::PhantomData,
//         }
//     }
// }

// impl<'a, const NCHAN: usize, I> Iterator for FrameIterator<'a, NCHAN, I >
// where
//     I: StreamingIterator<Item = u8>
// {
//     type Item = [u8; NCHAN];

//     fn next(&mut self) -> Option<Self::Item> {    
//         // If it starts none, we're exhausted
//         if self.packet_iter.get().is_none() {
//             return None;
//         }        

//         while self.packet_iter.get().is_some() && self.pos + NCHAN > self.packet_iter.get().unwrap().len() {
//             // Read packets until we get one with a frame, or we hit the end of the packet iterator
//             self.packet_iter.advance();
//             self.pos = 0;
            
//         }

//         if self.packet_iter.get().is_none() {
//             return None;
//         }

//         let packet = self.packet_iter.get().unwrap();
        
//         let mut frame = [0u8; NCHAN];
//         for ch in 0..NCHAN {
//             frame[ch] = packet[self.pos + ch];
//         }
//         self.pos += NCHAN;
        
//         return Some(frame);   
//     }
// }

// pub struct FixedScale<S> {
//     source: S,
//     scale: i32,
// }

// impl<S> FixedScale<S> {
//     /// Scale is a 16.16 fixed point multiplier
//     pub fn new(source: S, scale: i32) -> Self {
//         Self { source, scale }
//     }
// }

// impl<S> Signal for FixedScale<S>
// where
//     S: Signal,
//     <<S as Signal>::Frame as Frame>::Sample: Duplex<i32>,
// {
//     type Frame = S::Frame;

//     fn next(&mut self) -> Self::Frame {
//         let frame_in = self.source.next();
//         frame_in.map(|s| {
//             let x: i64 = s.to_sample::<i32>() as i64;
//             let y: i32 = ((x.saturating_mul(self.scale as i64)) / 65536) as i32;
//             y.to_sample::<<Self::Frame as Frame>::Sample>()
//         })
//     }

//     fn is_exhausted(&self) -> bool {
//         self.source.is_exhausted()
//     }
// }

// pub struct BiquadFilter<S, const NCHAN: usize> 
// where
//     S: Signal,
// {
//     source: S,
//     filters: [biquad::DirectForm2Transposed<f32>; NCHAN],
// }

// impl<S, const NCHAN: usize> BiquadFilter<S, NCHAN> 
// where
//     S: Signal,
// {
//     pub fn new(source: S, coeffs: biquad::Coefficients<f32>) -> Self {

//         let filters = [biquad::DirectForm2Transposed::<f32>::new(coeffs); NCHAN];
//         Self { source, filters}
//     }
// }

// impl <S, const NCHAN: usize> Signal for BiquadFilter<S, NCHAN> 
// where
//     S: Signal,
//     <<S as Signal>::Frame as Frame>::Sample: Duplex<f32>,
// {
//     type Frame = S::Frame;

//     fn next(&mut self) -> Self::Frame {
//         use biquad::Biquad;
        
//         let fin = self.source.next();
//         Self::Frame::from_fn(|ch| {
//             let x_f = fin.channel(ch).unwrap().to_sample::<f32>();
//             self.filters[ch].run(x_f).to_sample::<<Self::Frame as Frame>::Sample>()
//         })
//     }

//     fn is_exhausted(&self) -> bool {
//         self.source.is_exhausted()
//     }
// }


// // #[cfg(test)]
// // mod tests {
// //     use super::*;
// // //     #[test]
// // //     fn test_frame_iterator() {
// // //         let packet1: Vec<u8> = vec!(0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15);
// // //         let packet2: Vec<u8> = vec!(20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35);
// // //         let packets = vec!(&packet1[..], &packet2[..]);
// // //         let iterator = FrameIterator::<6, _>::new(packets.into_iter());
// // //         let output: Vec<[u8; 6]> = iterator.collect();
// // //         assert_eq!(output.len(), 4);
// // //         assert_eq!(output[0], [0, 1, 2, 3, 4, 5]);
// // //         assert_eq!(output[1], [10, 11, 12, 13, 14, 15]);
// // //         assert_eq!(output[2], [20, 21, 22, 23, 24, 25]);
// // //         assert_eq!(output[3], [30, 31, 32, 33, 34, 35]);
// // //     }

// //     #[test]
// //     fn test_fixed_scale() {
// //         let signal_in = [-10000, 100, 32767];
// //         let scaled = FixedScale::new(dasp::signal::from_iter(signal_in), 32768);
// //         let output = scaled.until_exhausted().collect::<Vec<i16>>();
// //         for i in 0..signal_in.len() {
// //             assert_eq!(output[i], signal_in[i] / 2)
// //         }

// //     }

// // }

