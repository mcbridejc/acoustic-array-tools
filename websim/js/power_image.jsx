
import React from 'react';

import Box from '@mui/material/Box';
import Checkbox from '@mui/material/Checkbox';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import Grid from '@mui/material/Grid';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';

import SquareIcon from '@mui/icons-material/Square';

import chroma from 'chroma-js';

const bilinear_interp = (power, width, x, y) => {
    let imageidx = (x, y) => {
        let i = x + y * width;
        return i
    }

    let height = power.length / width;

    // Get corner x/y coordinates, repeating the edge pixel for any pixel going off the edge
    let x0 = Math.floor(x);
    let x1 = Math.ceil(x);
    x1 = x1 > width - 1 ? width - 1 : x1;
    let y0 = Math.floor(y);
    let y1 = Math.ceil(y);
    y1 = y1 > height - 1? height - 1 : y1;

    // Get the four corner pixel values
    let p00 = power[imageidx(x0, y0)];
    let p01 = power[imageidx(x0, y1)];
    let p10 = power[imageidx(x1, y0)];
    let p11 = power[imageidx(x1, y1)];

    let kx1 = x - Math.floor(x);
    let ky1 = y - Math.floor(y);
    let kx0 = 1.0 - kx1;
    let ky0 = 1.0 - ky1;
    let p = p00*kx0*ky0 + p01*kx0*ky1 + p10*kx1*ky0 + p11*kx1*ky1;
    return p;
}


const scale_colors = [
  chroma.gl(0.0, 0.0, 0.1),
  chroma.gl(0.48, 0.8, 0.37),
  chroma.gl(0.98, 1.0, 0.64),
  chroma.gl(1.0, 0.5, 0.022),
  chroma.gl(0.98, 0.1, 0.08),
  chroma.gl(0.75, 0.0, 0.08)
];
const color_scale = chroma.scale(scale_colors);

const displayRangeDb = 6.0;

// Defines the total x/y span of the calculated power focal plane in meters
// Must match the one hardcoded in lib.rs. Should definitely fix this in one place...
const imageSpan = 1.8;
// Size of canvas in px
const canvasSize = 500;

function real2px(p) {
  const x = p[0];
  const y = p[1];
  return [(x + imageSpan / 2) * canvasSize / imageSpan, (y + imageSpan / 2) * canvasSize / imageSpan];
}

function px2real(p) {
  const x = p[0];
  const y = p[1];
  return [x * imageSpan / canvasSize - imageSpan / 2, y * imageSpan / canvasSize - imageSpan / 2];
}

function Legend(props) {
  return <div>
    {
      props.segments.map((s, i) => {
        return <span key={i} sx={{width: "30px"}}>
          <span>{s.label}</span><SquareIcon sx={{color: s.color}} />
        </span>;
      })
    }
  </div>
}



export default function PowerImage(props)  {
    let {power, width, sourceLocation} = props;

    let [range, setRange] = React.useState(displayRangeDb);
    let [absoluteUnits, setAbsoluteUnits] = React.useState(false);
    let [hoverPos, setHoverPos] = React.useState([0, 0]);
    let canvasRef = React.useRef(null);

    let pmax = Math.max(...power);
    let pmin = Math.min(...power);
    if (pmax - pmin > range) {
      pmin = pmax - range;
    } else {
      pmax = pmin + range;
    }

    let legend_points = scale_colors.map((c, i) => {
      let level = pmin + (pmax - pmin) * (i ) / (scale_colors.length-1);
      if(!absoluteUnits) {
        level -= pmax;
      }
      level = Math.round(level * 10) / 10;
      return { label: `${level.toString().padStart(6)}dB`, color: c.hex() };
    });

    React.useEffect(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      let w = canvas.width;
      let h = canvas.height;

      if(power && width) {
        let imgData = ctx.createImageData(w, h);

        let height = Math.ceil(power.length / width);

        for(let row=0; row<height; row++) {
          for(let col=0; col<width; col++) {
            const p = power[row*width + col];
            let coloridx = (p - pmin) / (pmax - pmin);
            let color = color_scale(coloridx).hex();
            ctx.fillStyle = color;
            ctx.fillRect(
              Math.floor(col * w / width),
              Math.floor(row * h / height),
              Math.ceil(w / width),
              Math.ceil(h / height),
            );
          }
        }

        // One can also draw the image with interpolation as below for smoother display, but I think
        // its better with nearest neighbor.

        // for(let row=0; row<h; row++) {
        //     for(let col=0; col<w; col++) {
        //         let p = bilinear_interp(power, width, col * width / w, row * height / h);
        //         let coloridx = (p - pmin) / (pmax - pmin);
        //         let color = color_scale(coloridx);
        //         const imgIdx = (row * w + col) * 4;
        //         const rgb = color.rgb();
        //         imgData.data[imgIdx + 0] = rgb[0];
        //         imgData.data[imgIdx + 1] = rgb[1];
        //         imgData.data[imgIdx + 2] = rgb[2];
        //         imgData.data[imgIdx + 3] = 255;
        //     }
        // }

        // ctx.putImageData(imgData, 0, 0);

        ctx.strokeStyle = "#ffffff"
        ctx.lineWidth = 2
        ctx.beginPath();
        let sourcePx = real2px(sourceLocation);
        ctx.arc(sourcePx[0], sourcePx[1], 20, 0, Math.PI * 2, true);
        ctx.stroke();



      }
    });

    const onRangeChange = (e, value) => {
      setRange(value);
    }

    const onCanvasClick = (event) => {
      const rect = event.target.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      if (props.onSourceChange) {
        props.onSourceChange(px2real([x, y]));
      }
    }

    const handleMouseMove = (e) => {
      let rect = e.target.getBoundingClientRect()
      let x = e.clientX - rect.left;
      let y = e.clientY - rect.top;
      //setHoverPos(px2real([x, y]));
      let pos = px2real([x, y])
      // Convert to mm and round to 0.1mm
      pos[0] = Math.round(pos[0]*1000*10) / 10;
      pos[1] = Math.round(pos[1]*1000*10) / 10;
      setHoverPos(pos);
    };

    const handleAbsoluteChange = (e) => {
      console.log(e.target.checked);
      setAbsoluteUnits(e.target.checked);
    }

    return <div>
      <Grid container>
        <Grid item>
          <FormControl>
            <FormGroup row>
              <Tooltip title="Choose colorbar power units to be either absolute, or relative to peak power in each image">
              <FormControlLabel
                value="absolute"
                control={<Checkbox value={absoluteUnits} onChange={handleAbsoluteChange} />}
                label="Absolute"
                labelPlacement="end"
              />
              </Tooltip>
            </FormGroup>
          </FormControl>
          <Legend segments={legend_points} />
          <Tooltip followCursor={true} title={`${hoverPos}`}>
            <canvas ref={canvasRef} width={canvasSize} height={canvasSize} onMouseMove={handleMouseMove} onClick={onCanvasClick} />
          </Tooltip>
        </Grid>
        <Grid item>
          <Box width={20} height={canvasSize}>
            <Tooltip title="Adjust displayed range/sensitivity (in dB)">
              <Slider
                orientation="vertical"
                getAriaLabel={() => 'Color Range'}
                value={range}
                onChange={onRangeChange}
                valueLabelDisplay="auto"
                min={1.0}
                max={20.0}
                step={0.1}
              />
            </Tooltip>
          </Box>
        </Grid>
      </Grid>

    </div>
}


