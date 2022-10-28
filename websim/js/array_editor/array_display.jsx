import React from 'react';
import PropTypes from 'prop-types';

import Paper from '@mui/material/Paper';
import Tooltip from '@mui/material/Tooltip';

import Draggable from 'react-draggable';

import CircleIcon from '@mui/icons-material/Circle';


export default function ArrayDisplay(props) {
  let {positions, scale, width, height} = props;

  let [hoverPos, setHoverPos] = React.useState([0, 0]);

  let real2px = (p) => {
    let x = p[0];
    let y = p[1];

    return [
      width * (0.5 + x / scale),
      height * (0.5 + y / scale)
    ];
  }

  let px2real = (p) => {
    let x = p[0];
    let y = p[1];

    x = (x / width - 0.5) * scale;
    y = (y / height - 0.5) * scale;
    return [
      Math.round(x*100)/100.0,
      Math.round(y*100)/100.0
    ];
  }

  let handleDrag = (i) => (e, data) => {
    if (props.onChange) {
      props.onChange(i, px2real([data.x, data.y]), false);
    }
  }

  let handleStop = (i) => (e, data) => {
    if (props.onChange) {
      props.onChange(i, px2real([data.x, data.y]), true);
    }
  }

  const handleClick = (e) => {
    // Filter events from child clicks
    if (e.target !== e.currentTarget) {
      return;
    }
    let rect = e.target.getBoundingClientRect()
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    if(e.shiftKey) {
      if (props.onAddPoint) {
        props.onAddPoint(px2real([x - 6.0, y - 6.0]));
      }
    }
  };

  const handleMicClick = (i) => (e) => {
    if(e.ctrlKey) {
      if (props.onRemovePoint) {
        console.log("Removing mic ", i);
        props.onRemovePoint(i);
      }
    }
  };

  const handleMouseMove = (e) => {
    let rect = e.target.getBoundingClientRect()
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    //setHoverPos(px2real([x, y]));
    setHoverPos(px2real([x, y]));
  };

  const changeHighlight = (i) => (e) => {
    if (props.onHighlight) {
      props.onHighlight(i);
    }
  };

  return <Paper>
    <Tooltip followCursor={true} title={`${hoverPos}`}>
    <div style={{height: height, width: width, position: 'relative'}} onMouseMove={handleMouseMove} onClick={handleClick}>
      {
        positions.map((p, i) => {
          p = real2px(p);
          return <Draggable position={{x: p[0], y: p[1]}}  onDrag={handleDrag(i)} onStop={handleStop(i)} key={i} bounds="parent">

            <div style={{width: "fit-content", position: 'absolute'}}>
              <CircleIcon
                fontSize="12px"
                sx={{ color: props.highlight == i ? "red" : "black" }}
                onMouseEnter={changeHighlight(i)}
                onMouseLeave={changeHighlight(null)}
                onClick={handleMicClick(i)}
              />
            </div>
          </Draggable>
        })
      }

    </div>
    </Tooltip>
  </Paper>
}


ArrayDisplay.propTypes = {
  height: PropTypes.number.isRequired,
  width: PropTypes.number.isRequired,
  scale: PropTypes.number.isRequired,
  onChange: PropTypes.func,

}