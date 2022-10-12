use plotters::prelude::*;
use plotters_cairo::CairoBackend;
use gtk::prelude::*;
use gtk::DrawingArea;
use std::rc::Rc;
use std::cell::RefCell;

struct InnerData {
    draw_area: DrawingArea,
    caption: String,
    ymin: f32,
    ymax: f32,
    xmin: f32,
    xmax: f32,
    margin: f32,
    color: RGBColor,
    line: Vec<(f32, f32)>,
    mesh: bool,
}

pub struct LinePlot {
    data: Rc<RefCell<InnerData>>
}

impl LinePlot {
    pub fn new(draw_area: DrawingArea, caption: &str) -> Self {
        let data = Rc::new(RefCell::new(InnerData {
            draw_area,
            caption: caption.to_string(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 0.0,
            margin: 10.0,
            color: BLACK,
            line: Vec::new(),
            mesh: false,
        }));
        let result = Self { data };
        result.connect()
    }

    fn connect(self) -> Self {
        let data = self.data.clone();
        self.data.borrow().draw_area.connect_draw(move |widget, cr| {
            let data = data.borrow();
            let w = widget.allocated_width();
            let h = widget.allocated_height();
            let backend = CairoBackend::new(cr, (w as u32, h as u32)).unwrap();
    
            let root = backend.into_drawing_area();
    
            root.fill(&WHITE).unwrap();
            let mut chart = ChartBuilder::on(&root).caption("RMS envelope", ("sans-serif", 22).into_font())
            .margin(data.margin)
            .y_label_area_size(60)
            .x_label_area_size(30)
            .build_cartesian_2d(data.xmin..data.xmax, data.ymin..data.ymax).unwrap();
            
            if data.mesh {
                chart.configure_mesh().draw().unwrap();
            }
    
            chart.draw_series(LineSeries::new(
                data.line.clone(),
                &data.color,
            )).unwrap();
    
            root.present().unwrap();
            Inhibit(false)
        });
        self
    }

    pub fn update(&self, line_data: Vec<(f32, f32)>) -> &Self {
        let mut data = self.data.borrow_mut();
        data.line = line_data;
        data.draw_area.queue_draw();
        self
    }

    pub fn set_xlim(&self, xmin: f32, xmax: f32) -> &Self {
        let mut data = self.data.borrow_mut();
        data.xmin = xmin;
        data.xmax = xmax;
        self
    }

    pub fn set_ylim(&self, ymin: f32, ymax: f32) -> &Self {
        let mut data = self.data.borrow_mut();
        data.ymin = ymin;
        data.ymax = ymax;
        self
    }

    pub fn set_color(&self, color: RGBColor) -> &Self {
        let mut data = self.data.borrow_mut();
        data.color = color;
        self
    }

    pub fn set_mesh(&self, mesh: bool) -> &Self {
        let mut data = self.data.borrow_mut();
        data.mesh = mesh;
        self
    }
}