use gtk::prelude::*;
use gtk::glib;
use plotters::prelude::*;
use plotters_cairo::CairoBackend;
use process::{processors};

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::thread;

use std::time::Duration;

use crossbeam_channel::{unbounded, Sender};

mod udp_receiver;
mod process;


const GLADE_UI_SOURCE: &'static str = include_str!("app.glade");

fn build_ui(app: &gtk::Application, state: Arc<Mutex<GuiState>>) {
    let builder = gtk::Builder::from_string(GLADE_UI_SOURCE);

    let window = builder.object::<gtk::Window>("MainWindow").unwrap();

    window.set_title("Acoustic Monitor");

    window.set_application(Some(app));

    let rms_drawing_area: gtk::DrawingArea = builder.object("MainDrawingArea").unwrap();
    let spectrum_drawing_area: gtk::DrawingArea = builder.object("SpectrumDrawingArea").unwrap();
    let az_drawing_area: gtk::DrawingArea = builder.object("AzimuthDrawingArea").unwrap();
    let waterfall_drawing_area: gtk::DrawingArea = builder.object("WaterfallDrawingArea").unwrap();
    let waterfall_radio_button: gtk::RadioButton = builder.object("WaterfallRadioButton").unwrap();
    //let image_radio_button: gtk::RadioButton = builder.object("ImageRadioButton").unwrap();
    //let low_freq_scale: gtk::Scale = builder.object("LowFreqScale").unwrap();

    let pdm_queue_label: gtk::Label = builder.object("PdmQueueLabel").unwrap();
    let dropped_packets_label: gtk::Label = builder.object("DroppedPacketsLabel").unwrap();
    let overruns_label: gtk::Label = builder.object("OverrunsLabel").unwrap();
    let state_timeout = state.clone();
    let state_draw = state.clone();

    rms_drawing_area.connect_draw(move |widget, cr| {
        let w = widget.allocated_width();
        let h = widget.allocated_height();
        let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();

        let root = backend.into_drawing_area();
        let s = state_draw.lock().unwrap().clone();

        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root).caption("RMS envelope", ("sans-serif", 22).into_font())
        .margin(10)
        .y_label_area_size(60)
        .build_cartesian_2d(-0f32..1f32, -80f32..0f32).unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(LineSeries::new(
            (0..s.rms_series.len()).map(|i| (i as f32 / s.rms_series.len() as f32, s.rms_series[i])),
            &RED,
        )).unwrap();

        root.present().unwrap();

        Inhibit(false)
    });

    let state_spectrum = state.clone();
    spectrum_drawing_area.connect_draw(move |widget, cr| {
        let w = widget.allocated_width();
        let h = widget.allocated_height();
        let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();

        let root = backend.into_drawing_area();
        let s = state_spectrum.lock().unwrap().clone();

        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root).caption("Spectrum", ("sans-serif", 22).into_font())
        .margin(10)
        .y_label_area_size(60)
        .x_label_area_size(30)
        .build_cartesian_2d(0f32..12e3f32, -120f32..-20f32).unwrap();

        chart.configure_mesh().draw().unwrap();

        const FNYQ: f32 = 12e3;
        chart.draw_series(LineSeries::new(
            (0..s.avg_spectrum.len()).map(|i| (i as f32 * FNYQ / s.avg_spectrum.len() as f32, s.avg_spectrum[i])),
            &GREEN,
        )).unwrap();

        Inhibit(false)
    });

    let state_az = state.clone();
    az_drawing_area.connect_draw(move |widget, cr| {
        let w = widget.allocated_width();
        let h = widget.allocated_height();
        let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();

        let root = backend.into_drawing_area();
        let s = state_az.lock().unwrap().clone();

        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 60u32)
        .set_label_area_size(LabelAreaPosition::Bottom, 60u32)
        .caption("Focused Power", ("sans-serif", 22))
        .build_cartesian_2d(0f32..360f32, 0f32..12f32).unwrap();

        chart.configure_mesh().draw().unwrap();

        if s.az_history.len() < 5 {
            return Inhibit(false);
        }
        let mut pwr: Vec<f32> = vec![0.0; s.az_power.len()];
        for row in &s.az_history[s.az_history.len() - 5..] {
            for i in 0..row.len() {
                pwr[i] += row[i];
            }
        }
        for i in 0..pwr.len() {
            pwr[i] /= 5.;
        }

        let mut min = f32::INFINITY;
        for p in &pwr {
            if *p < min {
                min = *p;
            }
        }
        chart.draw_series(
            AreaSeries::new(
                (0..pwr.len()).map(|i| (i as f32 * 360.0 / pwr.len() as f32, pwr[i] - min)),
                0.0,
                &BLUE,
            ).border_style(&RED)
        ).unwrap();

        Inhibit(false)
    });

    use palette::{Gradient, LinSrgb};

    // let gradient = Gradient::from([
    //     (0.0, LinSrgb::new(0.00f32, 0.05, 0.20)),
    //     (0.5, LinSrgb::new(0.70, 0.10, 0.20)),
    //     (1.0, LinSrgb::new(0.95, 0.90, 0.30)),
    // ]);

    let gradient = Gradient::from([
        (0.0, LinSrgb::new(0.0f32, 0.0, 0.1)),
        (0.2, LinSrgb::new(0.48f32, 0.8, 0.37)),
        (0.4, LinSrgb::new(0.98f32, 1.0, 0.64)),
        (0.6, LinSrgb::new(1.0f32, 0.5, 0.22)),
        (0.8, LinSrgb::new(0.98f32, 0.1, 0.08)),
        (1.0, LinSrgb::new(0.75f32, 0.0, 0.08)),
    ]);

    let color_map: Vec<_> = gradient.take(100).collect();

    let state_waterfall = state.clone();
    waterfall_drawing_area.connect_draw(move |widget, cr| {
        let w = widget.allocated_width();
        let h = widget.allocated_height();
        let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();

        let root = backend.into_drawing_area();
        let s = state_waterfall.lock().unwrap().clone();

        if s.az_history.len() < 1 {
            return Inhibit(false);
        }

        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 60u32)
        .set_label_area_size(LabelAreaPosition::Bottom, 60u32)
        .caption("Waterfall", ("sans-serif", 22))
        .build_cartesian_2d(0f32..1f32, 0f32..1f32).unwrap();

        if waterfall_radio_button.is_active() {
            let cells = root.split_evenly((100, 100));
            //let cells = root.split_evenly((s.az_history.len(), s.az_history[0].len()));

            let plotting_area = chart.plotting_area();
            let mut cell_number = 0;
            for row in 0..s.az_history.len() {
                let spectrum = &s.az_history[row];
                let mut min = f32::INFINITY;
                for s in spectrum {
                    if *s < min {
                        min = *s;
                    }
                }
                for value in spectrum {
                    let mut color_idx = ((value - min) * 100.0 / 6.0) as usize;
                    if color_idx > 99 {
                        color_idx = 99;
                    }
                    let c = color_map[color_idx].into_components();
                    let cu8 = ((c.0 * 255.) as u8, (c.1 * 255.) as u8,(c.2 * 255.) as u8);
                    cells[cell_number].fill(&RGBColor(cu8.0, cu8.1, cu8.2)).unwrap();
                    cell_number += 1;
                }
            }
        } else {
            let cells = root.split_evenly((20, 20));

            let mut vmax = -f32::INFINITY;
            let mut vmin = f32::INFINITY;
            for v in &s.image_power {
                if *v > vmax {
                    vmax = *v;
                }
                if *v < vmin {
                    vmin = *v;
                }
            }
            if vmax - vmin < 5.0 {
                vmax = vmin + 5.0
            }

            for (cell, value) in cells.iter().zip(s.image_power) {
                let mut color_idx = ((value - vmin) * 100.0 / (vmax - vmin)) as usize;
                if color_idx > 99 {
                    color_idx = 99;
                }
                let c = color_map[color_idx].into_components();
                let cu8 = ((c.0 * 255.) as u8, (c.1 * 255.) as u8,(c.2 * 255.) as u8);
                cell.fill(&RGBColor(cu8.0, cu8.1, cu8.2)).unwrap();
            }
        }

        root.present().unwrap();

        Inhibit(false)
    });

    glib::source::timeout_add_local(Duration::from_millis(25), move || {
        let s = state_timeout.lock().unwrap().clone();

        pdm_queue_label.set_text(format!("{}", s.pdm_level).as_str());
        dropped_packets_label.set_text(format!("{}", s.dropped_packets).as_str());
        overruns_label.set_text(format!("{}", s.overruns).as_str());

        rms_drawing_area.queue_draw();
        spectrum_drawing_area.queue_draw();
        az_drawing_area.queue_draw();
        waterfall_drawing_area.queue_draw();
        Continue(true)
    });

    window.show_all();
}

#[derive(Clone, Default, Debug)]
struct GuiState {
    rms_series: Vec<f32>,
    avg_spectrum: Vec<f32>,
    az_power: Vec<f32>,
    az_history: Vec<Vec<f32>>,
    image_power: Vec<f32>,
    pdm_level: u32,
    dropped_packets: u32,
    overruns: u32,
}

use heapless::{
    pool,
    pool::singleton::Pool
};


fn main() {

    let (udp_tx, udp_rx) = unbounded();

    let break_signal = Arc::new(AtomicBool::new(false));

    let udp_thread = thread::Builder::new()
    .name("udp_rx".to_string())
    .spawn(move || {
        udp_receiver::udp_rx_task(break_signal.clone(), udp_tx);
    });

    let gui_state = Arc::new(Mutex::new(GuiState::default()));

    let gui_state_clone = gui_state.clone();

    let gui_thread = thread::Builder::new()
    .name("gui".to_string())
    .spawn(move || {
        let app = gtk::Application::new(
            Some("acoustic.beamforming.display"),
            Default::default(),
        );
        app.connect_activate(move |app| {
            build_ui(app, gui_state_clone.clone());
        });
        app.run()
    }).unwrap();

    let (mut pdm_process, mut post_process) = processors();




    let gui_state_pdm = gui_state.clone();
    let _pdm_thread = thread::Builder::new()
    .name("pdm".to_string())
    .spawn(move || {

        let wav_spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writers = Vec::new();
        for i in 0..process::NUM_CHANNELS {
            let filename = format!("{}_{}.wav", "capture", i);
            writers.push(hound::WavWriter::create(filename, wav_spec).unwrap());
        }

        pdm_process.add_hook(Box::new(move |buf| {
            for ch in 0..process::NUM_CHANNELS {
                let data_box = buf.pcm[ch].as_ref().unwrap();
                let data = unsafe { data_box.assume_init_ref() };
                for sample in data {
                    writers[ch].write_sample((sample * 32767.0) as i16).unwrap();
                }
            }
        }));

        let mut last_seq_id = 0u8;
        loop {
            let packet = udp_rx.recv().unwrap();
            let seq_id = packet[packet.len() - 2];
            let missed_packet = seq_id != last_seq_id.overflowing_add(1).0;
            let flags = packet[packet.len() - 1];
            last_seq_id = seq_id;

            pdm_process.push_pdm_chunk(&packet[0..packet.len()-2]);
            {
                let mut s = gui_state_pdm.lock().unwrap();
                if missed_packet {
                    s.dropped_packets += 1;
                }
                if flags != 0 {
                    s.overruns += 1;
                }

                s.pdm_level = udp_rx.len() as u32;
            }
        }
    });

    const LOW_FREQ: f32 = 6000.0;
    const HIGH_FREQ: f32 = 8000.0;
    const ACTIVITY_THRESHOLD: f32 = -60.0;

    let gui_state_post = gui_state.clone();
    let _post_thread = thread::Builder::new()
    .name("post".to_string())
    .spawn(move || {


        loop {

            let (mut az_powers, image_powers) = post_process.run(LOW_FREQ, HIGH_FREQ);




            if *post_process.rms_series.last().unwrap() < ACTIVITY_THRESHOLD {
                az_powers.fill(-80.0);
            }
            {
                let mut s = gui_state_post.lock().unwrap();
                s.rms_series = post_process.rms_series.clone();
                s.avg_spectrum = Vec::from(post_process.latest_avg_spectrum);
                s.az_power = Vec::from(az_powers);
                s.az_history.push(Vec::from(az_powers));
                if s.az_history.len() > 100 {
                    s.az_history.remove(0);
                }

                if s.image_power.len() == image_powers.len() {
                    for i in 0..s.image_power.len() {
                        s.image_power[i] = s.image_power[i] * 0.85 + image_powers[i] * 0.15;
                    }
                } else {
                    s.image_power = Vec::from(image_powers);
                }
            }


        }
    });


    gui_thread.join().unwrap();


}
