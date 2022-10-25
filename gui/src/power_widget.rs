
use palette::{
    Gradient,
    LinSrgb,
    Srgb,
    encoding::Linear,
    rgb::Rgb,
};
use plotters::prelude::*;
use plotters_cairo::CairoBackend;
use gtk::prelude::*;
use gtk::DrawingArea;
use std::rc::Rc;
use std::cell::RefCell;


struct InnerData {
    draw_area: gtk::DrawingArea,
    data: Vec<f32>,
    waterfall: bool,
    stride: usize,
}
/// A widget for displaying power image or waterfall on a gtk drawing area
pub struct PowerDisplay {
    data: Rc<RefCell<InnerData>>,
}

impl PowerDisplay {
    pub fn new(draw_area: gtk::DrawingArea) -> Self {


        // let gradient = Gradient::from([
        //     (0.0, LinSrgb::new(0.00f32, 0.05, 0.20)),
        //     (0.5, LinSrgb::new(0.70, 0.10, 0.20)),
        //     (1.0, LinSrgb::new(0.95, 0.90, 0.30)),
        // ]);
    
        
        let result = Self {
            data: Rc::new(RefCell::new(InnerData {
                draw_area,
                data: Vec::new(),
                waterfall: false,
                stride: 0,
            }))
        };
        result.connect()
    }

    fn connect(self) -> Self {
        let data = self.data.clone();
        let gradient = Gradient::from([
            (0.0, LinSrgb::new(0.0f32, 0.0, 0.1)),
            (0.2, LinSrgb::new(0.48f32, 0.8, 0.37)),
            (0.4, LinSrgb::new(0.98f32, 1.0, 0.64)),
            (0.6, LinSrgb::new(1.0f32, 0.5, 0.22)),
            (0.8, LinSrgb::new(0.98f32, 0.1, 0.08)),
            (1.0, LinSrgb::new(0.75f32, 0.0, 0.08)),
        ]);
    
        let color_map: Vec<_> = gradient.take(100).collect();
    
        self.data.borrow().draw_area.connect_draw(move |widget, cr| {
            let data = data.borrow();
            let w = widget.allocated_width();
            let h = widget.allocated_height();
            let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();
    
            let root = backend.into_drawing_area();
    
            root.fill(&WHITE).unwrap();
            
            if data.waterfall && data.stride > 0 {
                let cells = root.split_evenly((100, 100));
    
                let mut cell_number = 0;
                let nrows = data.data.len() / data.stride;
                for row in 0..nrows {
                    let spectrum = &data.data[row*data.stride..(row+1)*data.stride];
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
            } else if data.stride > 0 {
                let nrows = data.data.len() / data.stride;
                if nrows > 0 && data.stride > 0 {
                
                    let cells = root.split_evenly((data.stride, nrows));
        
                    let mut vmax = -f32::INFINITY;
                    let mut vmin = f32::INFINITY;
                    for v in &data.data {
                        if *v > vmax {
                            vmax = *v;
                        }
                        if *v < vmin {
                            vmin = *v;
                        }
                    }
                    
                    if vmax - vmin > 6.0 {
                        vmin = vmax - 6.0;
                    } else {
                        vmax = vmin + 6.0;
                    }
        
                    for (cell, value) in cells.iter().zip(&data.data) {
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
            }
            Inhibit(false)
        });
        self
    }
    /// Display a time-series of 1-dimensional power vectors in "waterfall" form
    pub fn display_waterfall(&self, waterfall: &Vec<Vec<f32>>) {
        let mut data = self.data.borrow_mut();
        data.waterfall = true;
        data.stride = if waterfall.len() > 0 { 
            waterfall[0].len() 
        } else { 
            0 
        };

        data.data.clear();
        for row in waterfall {
            data.data.extend_from_slice(row);
        }
        data.draw_area.queue_draw();
    }

    /// Display a 2D power vector as an image
    pub fn display_image(&self, image: &Vec<f32>, width: usize) {
        let mut data = self.data.borrow_mut();
        data.waterfall = false;
        data.stride = width;
        data.data.clear();
        data.data.extend_from_slice(image);
        data.draw_area.queue_draw();
    }
}




