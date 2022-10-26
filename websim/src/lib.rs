use wasm_bindgen::prelude::*;
use js_sys::Array;

use ndarray::{Array2, Axis};

use dsp::beamforming::{
    HeapBeamFormer,
    FftProcessor, BeamFormer,
};

use dsp::generation::WhiteNoiseSource;
use dsp::buffer::HeapSpectra;

const FSAMPLE: f32 = 24e3;
const WINDOW_SIZE: usize = 1024;
const NFFT: usize = WINDOW_SIZE / 2 + 1;
const GRID_RES: usize = 20;
const SPEED_OF_SOUND: f32 = 343.0;

pub fn make_grid_focal_points<const N: usize, const M: usize>(width: f32, z: f32) -> Array2<f32>
{
    // This feels terrible, but I cannot find a way to size the array without passing two generic arguments
    assert!(N*N == M);
    let mut points = Array2::zeros((M, 3));

    let mut i = 0;
    for iy in 0..N {
        let y = -width / 2.0 + width * iy as f32 / (N - 1) as f32;
        for ix in 0..N {
            let x = -width / 2.0 + width * ix as f32 / (N - 1) as f32;
            points[[i,0]] = x;
            points[[i,1]] = y;
            points[[i,2]] = z;
            i += 1;
        }
    }
    points
}

fn min(x: &[f32]) -> f32 {
    let mut min = f32::INFINITY;
    for val in x {
        if *val < min {
            min = *val;
        }
    }
    min
}

fn get_source(mics: &Array2<f32>, source_pos: [f32; 3]) -> WhiteNoiseSource {
    let mut src = Array2::zeros((1, 3));
    src[[0, 0]] = source_pos[0];
    src[[0, 1]] = source_pos[1];
    src[[0, 2]] = source_pos[2];
    let delta = mics - src;

    let sqr = delta.mapv(|a| a.powi(2));
    let mut dist = sqr.sum_axis(Axis(1)).mapv(|a| a.sqrt());

    let min_d = min(dist.as_slice().unwrap());
    dist -= min_d;

    // Convert from distance (meters) to delay (sample periods)
    let channel_delays: Vec<f32> = dist.iter().map(|d| FSAMPLE * d / SPEED_OF_SOUND).collect();

    WhiteNoiseSource::new(&channel_delays)
}

#[wasm_bindgen]
pub struct System {
    beamformer: HeapBeamFormer,
    mics: Array2<f32>,
    focal_points: Array2<f32>,
}

#[wasm_bindgen]
impl System {
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            beamformer: HeapBeamFormer::new(FSAMPLE),
            mics: Array2::zeros((0, 0)),
            focal_points: make_grid_focal_points::<GRID_RES, {GRID_RES*GRID_RES}>(2.0, 0.5),
        }
    }

    // mic_positions should be a javascript array of floats, with flattened 2d positions, i.e.
    // [p0_x, p0_y, p1_x, p1_y, ..., pn_x, pn_y].
    pub fn set_mics(&mut self, mic_positions: Array) {
        
        // Convert to rust array
        assert!(mic_positions.length() % 2 == 0);
        let n_mics = mic_positions.length() as usize / 2;

        let mut mics = Array2::zeros((n_mics, 3));
        for i in 0..n_mics {
            let x_mm = mic_positions.get(i as u32).as_f64().expect("Could not convert mic positions to f64");
            let y_mm = mic_positions.get((i+1) as u32).as_f64().expect("Could not convert mic positions to f64");

            mics[[i, 0]] = x_mm as f32 / 1000.0;
            mics[[i, 1]] = y_mm as f32 / 1000.0;
            mics[[i, 2]] = 0.0;
        }

        self.mics = mics;
        self.beamformer.setup(&self.mics, &self.focal_points, NFFT, FSAMPLE);
    }

    pub fn run(&mut self, x: f32, y: f32, z: f32, start_freq: f32, end_freq: f32) -> Array {

        let mut source = get_source(&self.mics, [x, y, z]);
        let mut samples = source.next(WINDOW_SIZE);

        let mut spectra = HeapSpectra::new(NFFT, self.mics.dim().0);

        let mut fft = FftProcessor::new(WINDOW_SIZE);

        futures::executor::block_on(fft.compute_ffts(&mut samples, &mut spectra));

        let mut power_out = vec![0.0; self.focal_points.dim().0];
        self.beamformer.compute_power(&mut spectra, &mut power_out, start_freq, end_freq);

        let ret = Array::new_with_length((GRID_RES * GRID_RES) as u32);
        for i in 0..GRID_RES*GRID_RES {
            ret.set(i as u32, JsValue::from_f64(power_out[i] as f64));
        }
        ret
    }
}

