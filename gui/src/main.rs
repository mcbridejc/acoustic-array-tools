//#![feature(future_poll_fn)]
#![feature(async_closure)]

use dsp::buffer::Spectra;
use gtk::prelude::*;
use gtk::glib;
use plotters::prelude::*;
use plotters_cairo::CairoBackend;
use process::{processors};

use std::sync::{
    Arc,
    atomic::{AtomicBool},
    Mutex,
};

use std::rc::Rc;
use std::thread;

use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded};

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
    let compass_drawing_area: gtk::DrawingArea = builder.object("CompassDrawingArea").unwrap();
    let waterfall_radio_button: gtk::RadioButton = builder.object("WaterfallRadioButton").unwrap();
    let low_freq_scale: Rc<gtk::Scale> = Rc::new(builder.object("LowFreqScale").unwrap());
    let high_freq_scale: Rc<gtk::Scale> = Rc::new(builder.object("HighFreqScale").unwrap());
    let process_time_scale: gtk::Scale = builder.object("ProcessTimeScale").unwrap();

    let pdm_queue_label: gtk::Label = builder.object("PdmQueueLabel").unwrap();
    let dropped_packets_label: gtk::Label = builder.object("DroppedPacketsLabel").unwrap();
    let overruns_label: gtk::Label = builder.object("OverrunsLabel").unwrap();
    let state_timeout = state.clone();
    let state_draw = state.clone();

    {
        let mut s = state.lock().unwrap();
        s.low_freq = low_freq_scale.value() as f32;
        s.high_freq = high_freq_scale.value() as f32;
        s.process_time = process_time_scale.value() as f32;
    }

    let state_low_freq = state.clone();
    let state_high_freq = state.clone();
    let low_freq_clone = low_freq_scale.clone();
    let high_freq_clone = high_freq_scale.clone();

    high_freq_scale.connect_value_changed(move |target| {
        // Update the state, and return the values, releasing the lock before updating the other scale
        let (low, high) = {
            let mut state = state_high_freq.lock().unwrap();
            state.high_freq = target.value() as f32;
            (state.low_freq, state.high_freq)
        };
        if high < low {
            low_freq_clone.set_value(high as f64);
        }
    });

    low_freq_scale.connect_value_changed(move |target| {
        let (low, high) = {
            let mut state = state_low_freq.lock().unwrap();
            state.low_freq = target.value() as f32;
            (state.low_freq, state.high_freq)
        };
        if low > high {
            high_freq_clone.set_value(low as f64);
        }
    });

    let state_process_time = state.clone();
    process_time_scale.connect_value_changed(move |target| {
        let mut s = state_process_time.lock().unwrap();
        s.process_time = target.value() as f32;
    });

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
        .build_cartesian_2d(0f32..12e3f32, -140f32..-40f32).unwrap();

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
        .caption("Azimuth Power", ("sans-serif", 22))
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

        if waterfall_radio_button.is_active() {
            let cells = root.split_evenly((100, 100));

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
            // if vmax - vmin < 5.0 {
            //     vmax = vmin + 5.0;
            // }
            // if vmax - vmin > 12.0 {
            //     vmin = vmax - 12.0;
            // }
            
            if vmax - vmin > 6.0 {
                vmin = vmax - 6.0;
            } else {
                vmax = vmin + 6.0;
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

    use image::{imageops::FilterType, ImageFormat};
    use std::io::BufReader;
    use std::fs::File;
    let state_compass = state.clone();
    let compass_image = image::load(
        BufReader::new(
            File::open("images/compass_background.jpg").map_err(|e| {
                eprintln!("Unable to open file plotters-doc-data.png, please make sure you have clone this repo with --recursive");
                e
            }).unwrap()),
        ImageFormat::Jpeg,
    ).unwrap();


    // let needle_image = image::load(
    //     BufReader::new(
    //         File::open("images/compass_needle.png").map_err(|e| {
    //             eprintln!("Unable to open file plotters-doc-data.png, please make sure you have clone this repo with --recursive");
    //             e
    //         }).unwrap()),
    //     ImageFormat::Png,
    // ).unwrap()


    compass_drawing_area.connect_draw(move |widget, cr| {
        let w = widget.allocated_width();
        let h = widget.allocated_height();
        // let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();

        // let root = backend.into_drawing_area();
        let s = state_compass.lock().unwrap().clone();

        // root.fill(&WHITE).unwrap();

        // let mut chart = ChartBuilder::on(&root)
        //     .margin(0)
        //     .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        //     .unwrap();
        // chart.configure_mesh().disable_mesh().draw().unwrap();
        // let background = compass_image.resize_exact(w as u32, h as u32, FilterType::Lanczos3);

        // let needle_image = raster::open("images/compass_needle.png").unwrap();
        // let needle = raster::transform::resize_exact(&mut needle_image, w, h);
        // let needle = raster::transform::rotate(&mut needle_image, s.look_angle as i32, raster::Color::rgba(0, 0, 0, 255)).unwrap();

        let cx = w as f64 / 2.;
        let cy = h as f64 / 2.;
        let r = if w > h { h } else { w } as f64 * 0.5;
        let b = r / 10.0;
        let rot = |p: (f64, f64), angle: f64| {
            let c = f64::cos(angle as f64);
            let s = f64::sin(angle as f64);
            (c * p.0 - s * p.1, s * p.0 + c * p.1)
        };
        let p1 = rot((0., -b), s.look_angle as f64);
        let p2 = rot((0., b), s.look_angle as f64);
        let p3 = rot((r, 0.), s.look_angle as f64);
        
        cr.move_to(cx, cy);
        cr.line_to(cx + p1.0, cy + p1.1);
        cr.line_to(cx + p2.0, cy + p2.1);
        cr.line_to(cx + p3.0, cy + p3.1);
        cr.line_to(cx + p1.0, cy + p1.1);
        cr.line_cap();

        cr.stroke().unwrap();
        // let elem: BitMapElement<_> = ((0.00, 1.0), image).into();
        // chart.draw_series(std::iter::once(elem)).unwrap();


        // let elem: BitMapElement<_> = ((0.00, 1.0), needle_image).into();
        // chart.draw_series(std::iter::once(elem)).unwrap();

        //root.present().unwrap();
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
        compass_drawing_area.queue_draw();
        Continue(true)
    });

    window.show_all();
}

#[derive(Clone, Default, Debug)]
struct GuiState {
    // Time series with history of RMS values in dB relative to full-scale
    rms_series: Vec<f32>,
    // FFT series with combined spectra for all channels
    avg_spectrum: Vec<f32>,
    // Total measured power at all azimuth focal points from latest time window
    az_power: Vec<f32>,
    // Running collection of az_power series over time, used to generate waterfal plot
    az_history: Vec<Vec<f32>>,
    // Total measured power at all image focal points from latest time window
    image_power: Vec<f32>,
    // Number of PDM blocks awaiting processing
    pdm_level: u32,
    // Running tally of missed UDP packets
    dropped_packets: u32,
    // Running tally of DMA overruns reported by the embedded software
    overruns: u32,
    // latest mannequin look angle command
    look_angle: f32,
    // The low end of the frequency window used for power calculation
    low_freq: f32,
    // The high end of the frequency window used for power calculation
    high_freq: f32,
    // The minimum time to process one audio window, in ms
    process_time: f32,
}

fn main() {
    let break_signal = Arc::new(AtomicBool::new(false));
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


    let (udp_tx, udp_rx) = unbounded();
    let (mut udp_collector, processor) = processors();

    let _udp_thread = thread::Builder::new()
    .name("udp_rx".to_string())
    .spawn(move || {

        udp_receiver::udp_rx_task(break_signal.clone(), udp_tx);
    });

    let gui_state_collect = gui_state.clone();
    let _udp_collect_thread = thread::Builder::new()
    .name("udp_collect".to_string())
    .spawn(move || {
        // let start = Instant::now();
        // let mut finalized = false;

        // let wav_spec = hound::WavSpec {
        //     channels: 1,
        //     sample_rate: 24000,
        //     bits_per_sample: 16,
        //     sample_format: hound::SampleFormat::Int,
        // };

        // let mut writers = Vec::new();
        // for i in 0..process::NUM_CHANNELS {
        //     let filename = format!("{}_{}.wav", "capture", i);
        //     writers.push(hound::WavWriter::create(filename, wav_spec).unwrap());
        // // }

        // pdm_process.add_hook(Box::new(move |buf| {
        //     let duration = Instant::now() - start;
        //     if duration > Duration::from_secs(5) {
        //         if !finalized {
        //             while writers.len() > 0 {
        //                 let w = writers.remove(0);
        //                 w.finalize().unwrap();
        //             }
        //             finalized = true;
        //         }
        //     } else {
        //         for ch in 0..process::NUM_CHANNELS {
        //             let data_box = buf.pcm[ch].as_ref().unwrap();
        //             let data = unsafe { data_box.assume_init_ref() };
        //             for sample in data {
        //                 writers[ch].write_sample((sample * 32767.0) as i16).unwrap();
        //             }
        //         }
        //     }
        // }));

        let mut last_seq_id = 0u8;
        loop {
            let packet = udp_rx.recv().unwrap();
            let seq_id = packet[packet.len() - 2];
            let missed_packet = seq_id != last_seq_id.overflowing_add(1).0;
            let flags = packet[packet.len() - 1];
            last_seq_id = seq_id;

            udp_collector.push_pdm_chunk(&packet[0..packet.len()-2]);
            {
                let mut s = gui_state_collect.lock().unwrap();
                if missed_packet {
                    s.dropped_packets += 1;
                }
                if flags != 0 {
                    s.overruns += 1;
                }

                s.pdm_level = udp_collector.len() as u32 + 1;
            }
        }
    });

    const ACTIVITY_THRESHOLD: f32 = -100.0;
    const ANGLE_OFFSET: f32 = 132.0;
    let mut port = serialport::new("/dev/ttyUSB0", 115200).open().ok();
    if port.is_none() {
        println!("Couldn't open /dev/ttyUSB0, so no motor commands will be sent");
    }

    let gui_state_proc = gui_state.clone();
    let _post_thread = thread::Builder::new()
    .name("processing".to_string())
    .spawn(move || {
        use futures::executor::LocalPool;
        use futures::task::{LocalSpawn};

        let mut pool = LocalPool::new();

        let processor_pre = Rc::new(processor);
        let processor_post = processor_pre.clone();

        let preproc = async move {
            loop {
                processor_pre.stage1(ACTIVITY_THRESHOLD).await;
            }
        };

        let postprocess = async move {

            let mut az_filter = process::AzFilter2::new();
            loop {
                let mut spectra = Spectra::blank();
                if processor_post.stage2(&mut spectra).await {

                    let start_time = Instant::now();
                    // A new spectra was completed
                    {
                        let mut s = gui_state_proc.lock().unwrap();
                        s.rms_series.push(spectra.rms);
                        if s.rms_series.len() > 200 {
                            s.rms_series.remove(0);
                        }
                    }

                    let (az_powers, image_powers) = if spectra.spectra.is_some() && spectra.rms > -55. {
                        // Get settings from the gui_state and release the lock
                        let (low_freq, high_freq) = {
                            let s = gui_state_proc.lock().unwrap();
                            (s.low_freq, s.high_freq)
                        };
                        processor_post.beamform_power(&spectra, low_freq, high_freq)
                    } else {
                        ([-100.0; process::N_AZ_POINTS], [-100.0; process::IMAGE_GRID_RES * process::IMAGE_GRID_RES])
                    };

                    if let Some(updated_angle) = az_filter.push(process::weighted_azimuth(&az_powers), spectra.rms) {
                        println!("Angle: {}", updated_angle * 180. / 3.14159);
                        let adjusted_angle = ANGLE_OFFSET - updated_angle * 180. / 3.14159;
                        let mut s = gui_state_proc.lock().unwrap();
                        s.look_angle = adjusted_angle;

                        //println!("{}", adjusted_angle as i32);
                        if port.is_some() {
                            port.as_mut().unwrap().write(format!("P {}\n", adjusted_angle as i32).as_bytes()).unwrap();
                        }
                    }

                    {
                        let mut s = gui_state_proc.lock().unwrap();
                        if let Some(avg_spec) = spectra.avg_mag() {
                            s.avg_spectrum = Vec::from(avg_spec);
                        }
                        s.az_power = Vec::from(az_powers);
                        s.az_history.push(Vec::from(az_powers));
                        if s.az_history.len() > 100 {
                            s.az_history.remove(0);
                        }

                        s.image_power = Vec::from(image_powers);
                        // if s.image_power.len() == image_powers.len() {
                        //     for i in 0..s.image_power.len() {
                        //         s.image_power[i] = s.image_power[i] * 0.85 + image_powers[i] * 0.15;
                        //     }
                        // } else {
                        //     s.image_power = Vec::from(image_powers);
                        // }
                    }

                    // If result had data to process, delay to simulate slower processing
                    if spectra.spectra.is_some() {
                        let process_time = {
                            let s = gui_state_proc.lock().unwrap();
                            Duration::from_millis(s.process_time as u64)
                        };
                        let elapsed = Instant::now() - start_time;

                        if elapsed < process_time {
                            let wait_time = process_time - elapsed;
                            async_std::task::sleep(wait_time).await;
                        }
                    }
                }

                embassy_futures::yield_now().await;
            }

        };

        let spawner = pool.spawner();
        spawner.spawn_local_obj(Box::new(preproc).into()).unwrap();
        spawner.spawn_local_obj(Box::new(postprocess).into()).unwrap();
        pool.run();
    });

    gui_thread.join().unwrap();
}

