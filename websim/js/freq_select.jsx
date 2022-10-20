import React from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';

function valuetext(value) {
  if(value < 1000) {
    return `${value}Hz`;
  } else {
    return `${value / 1000}kHz`;
  }
}

const minDistance = 100;
const maxValue = 10000;
const minValue = 100;

const marks = [
  {value: 100, label: "100Hz"},
  {value: 5000, label: "5kHz"},
  {value: 10000, label: "10kHz"},
];

export default function FreqSelect(props) {
  const [value, setValue] = React.useState([200, 1500]);



  const handleChange = (event, newValue, activeThumb) => {
    if (!Array.isArray(newValue)) {
      return;
    }

    if (newValue[1] - newValue[0] < minDistance) {
      if (activeThumb === 0) {
        const clamped = Math.min(newValue[0], maxValue - minDistance);
        setValue([clamped, clamped + minDistance]);
      } else {
        const clamped = Math.max(newValue[1], minValue + minDistance);
        setValue([clamped - minDistance, clamped]);
      }
    } else {
      setValue(newValue);
    }
    props.onChange(value)
  };

  return (
    <Box sx={{ margin: 1, width: 500 }}>
      <label>Beamforming frequency range: {valuetext(value[0])} to {valuetext(value[1])} </label>
      <Slider
        getAriaLabel={() => 'Frequency Range'}
        value={value}
        onChange={handleChange}
        valueLabelDisplay="auto"
        getAriaValueText={valuetext}
        min={minValue}
        max={maxValue}
        step={10}
        marks={marks}
        disableSwap
      />

    </Box>
  );
}
