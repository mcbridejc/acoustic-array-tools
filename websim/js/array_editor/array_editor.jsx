import React from 'react';
import PropTypes from 'prop-types';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid'

import ArrayDisplay from './array_display.jsx';
import LocationTable from './location_table.jsx';
import ArrayList from './array_list.jsx';


export default function ArrayEditor(props) {


  const defaultArrays = [
    {
      name: "Circle6",
      positions: [
        [-50.88, 29.4],
        [-50.88, -29.4],
        [50.88, 29.4],
        [0.0, 58.75],
        [50.88, -29.4],
        [0.0, -58.75],
      ]
    }
  ];

  const [arrays, setArrays] = React.useState(JSON.parse(localStorage.getItem('mic_arrays')) || defaultArrays);

  const [activeArrayIdx, setActiveArrayIdx] = React.useState(0);

  let activeArray = arrays[activeArrayIdx]; 

  // const [activeArray, setActiveArray] = React.useState(() => {
  //   if(props.onMicChange) {
  //     props.onMicChange(arrays[activeArrayIdx])
  //   }
  //   return arrays[activeArrayIdx];
  // });

  const [scale, setScale] = React.useState(1000);

  const [highlightIdx, setHighlightIdx] = React.useState(null);

  const updateScale = (positions, shrink) => {
    let max = Math.max(...positions.map((p) => { 
      return Math.max(
        Math.abs(p[0]),
        Math.abs(p[1])
      );
    }));
    
    if (max > 0.42 * scale || (shrink && (max < 0.38 * scale))) {
      let scale = max / 0.4;
      scale = scale < 50 ? 50 : scale;
      scale = scale > 10000 ? 10000 : scale;
      console.log("Scale change: ", scale);
      setScale(scale);
    }
  };

  /// TODO: Try to find a better way to edit these nested state objects in arrays? 
  /// This is just getting silly

  /// Handle position edits from the input boxes in the table
  const editMicPosition = (mic_idx, coord_idx, value) => {
    let activeCopy = {...activeArray};
    let updatePositions = [...activeCopy.positions];
    let updatePoint = [...updatePositions[mic_idx]];
    updatePoint[coord_idx] = Number(value);
    updatePositions[mic_idx] = updatePoint;
    activeCopy.positions = updatePositions;
    let arraysCopy = [...arrays];
    arraysCopy[activeArrayIdx] = activeCopy;
    setArrays(arraysCopy);
    updateScale(activeCopy.positions, false);
    if(props.onMicChange) {
      props.onMicChange(activeCopy);
    }
  };

  /// Handle positions changes from dragging on the ArrayDisplay
  const handleDisplayChange = (mic_idx, value, finished) => {
    let activeCopy = {...activeArray};
    let updatePositions = [...activeCopy.positions];
    updatePositions[mic_idx] = value;
    activeCopy.positions = updatePositions
    let arraysCopy = [...arrays];
    arraysCopy[activeArrayIdx] = activeCopy;
    setArrays(arraysCopy);
    updateScale(activeCopy.positions, finished);
    if(finished && props.onMicChange) {
      props.onMicChange(activeCopy);
    }
  }

  const handleDisplayAdd = (p) => {
    let activeCopy = {...activeArray};
    let updatePositions = [...activeCopy.positions];
    updatePositions.push(p);
    activeCopy.positions = updatePositions;
    let arraysCopy = [...arrays];
    arraysCopy[activeArrayIdx] = activeCopy;
    setArrays(arraysCopy);
    updateScale(activeCopy.positions, true);
    if(props.onMicChange) {
      props.onMicChange(activeCopy);
    }
  };

  const handleDisplayRemove = (mic_idx) => {
    let activeCopy = {...activeArray};
    let updatePositions = [...activeCopy.positions];
    updatePositions.splice(mic_idx, 1);
    activeCopy.positions = updatePositions
    let arraysCopy = [...arrays];
    arraysCopy[activeArrayIdx] = activeCopy;
    setArrays(arraysCopy);
    updateScale(activeCopy.positions, true);
    if(props.onMicChange) {
      props.onMicChange(activeCopy);
    }
  }

  const handleHighlight = (mic_idx) => {
    setHighlightIdx(mic_idx);
  }

  const handleSelectArray = (array_idx) => {
    setActiveArrayIdx(array_idx);
    updateScale(arrays[array_idx].positions, true);
    if(props.onMicChange) {
      props.onMicChange(arrays[array_idx]);
    }
  }

  const handleAddArray = (array_name) => {
    let arraysCopy = [...arrays];
    arraysCopy.push({
      name: array_name,
      positions: [],
    });
    setArrays(arraysCopy);
  }

  const handleImportJson = (new_positions) => {
    let activeCopy = {...activeArray};
    activeCopy.positions = new_positions
    let arraysCopy = [...arrays];
    arraysCopy[activeArrayIdx] = activeCopy;
    setArrays(arraysCopy);
    updateScale(activeCopy.positions, true);
    if(props.onMicChange) {
      props.onMicChange(activeCopy);
    }
  }

  const handleSaveArrays = () => {
    localStorage.setItem('mic_arrays', JSON.stringify(arrays));
  }

  return (

    <Box sx={{ borderRadius: "10px", width: "fit-content", bgcolor: '#E7EBF0' }}>
      <Grid container>
        <Grid item>
          <Box sx={{ margin: 1, height: 500, width: 500}}>
            <ArrayDisplay 
              positions={activeArray.positions} 
              height={500}
              width={500}
              scale={scale}
              highlight={highlightIdx}
              onRemovePoint={handleDisplayRemove}
              onAddPoint={handleDisplayAdd}
              onChange={handleDisplayChange} 
              onHighlight={handleHighlight}
            />
          </Box>
        </Grid>
        <Grid item>
          <Box sx={{ margin: 1, width: 250, height: 500, overflow: "hidden", overflowY: "auto" }}>
            <LocationTable positions={activeArray.positions} highlight={highlightIdx} onHighlight={handleHighlight} onEdit={editMicPosition} />
          </Box>
        </Grid>
        <Grid item>
          <Box sx={{ margin: 1, width: 250, height: 500, bgcolor: 'white', overflow: "hidden", overflowY: "auto" }}>
            <ArrayList 
              arrays={arrays}
              selectedIdx={activeArrayIdx}
              onSelect={handleSelectArray}
              onAdd={handleAddArray} 
              onSaveAll={handleSaveArrays}
              onImportJson={handleImportJson}
              
            />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
