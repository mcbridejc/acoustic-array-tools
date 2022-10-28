import React from 'react';
import PropTypes from 'prop-types';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid'
import Tooltip from '@mui/material/Tooltip';

import ArrayDisplay from './array_display.jsx';
import LocationTable from './location_table.jsx';
import ArrayList from './array_list.jsx';

export default function ArrayEditor(props) {
  const defaultArrays = [
    {
      name: "Circle D=118mm (6ch)",
      positions: [
        [-50.88, 29.4],
        [-50.88, -29.4],
        [50.88, 29.4],
        [0.0, 58.75],
        [50.88, -29.4],
        [0.0, -58.75],
      ]
    },
    {
      name: "BK WA-1764 (30ch)",
      positions: [
        [ -67.085, -110.213],
        [ -16.409, -118.322],
        [  21.765, -122.713],
        [  91.360,  -97.376],
        [ 110.617,  -51.092],
        [ 123.117,  -16.970],
        [  89.671,   16.813],
        [ 119.468,   50.259],
        [  80.887,   88.772],
        [  55.549,  109.718],
        [ -12.018,  126.610],
        [ -58.301,  105.664],
        [ -88.031,   84.380],
        [ -83.977,   38.096],
        [-126.207,   29.650],
        [-117.761,  -21.362],
        [-109.315,  -55.146],
        [ -62.693,  -67.646],
        [   0.481,  -67.646],
        [  47.103,  -80.822],
        [  34.603,  -29.808],
        [  63.995,  -21.362],
        [  38.657,   20.867],
        [  42.711,   54.650],
        [   9.265,   88.434],
        [ -12.018,   42.488],
        [ -41.747,   59.042],
        [ -45.801,    4.313],
        [ -67.423,  -21.362],
        [ -20.463,  -37.916]]
    },
    {
      name: "Circle D=300mm (16ch)",
      positions: [
        [0.0, -150.0],
        [57.403, -138.582],
        [106.066, -106.066],
        [138.582, -57.403],
        [150.0, -0.0],
        [138.582, 57.403],
        [106.066, 106.066],
        [57.403, 138.582],
        [0.0, 150.0],
        [-57.403, 138.582],
        [-106.066, 106.066],
        [-138.582, 57.403],
        [-150.0, 0.0],
        [-138.582, -57.403],
        [-106.066, -106.066],
        [-57.403, -138.582]
      ]
    },
    {
      name: "Cirlce D=300mm (64ch)",
      positions: [
        [0.0, -150.0],
        [14.703, -149.278],
        [29.264, -147.118],
        [43.543, -143.541],
        [57.403, -138.582],
        [70.71, -132.288],
        [83.336, -124.72],
        [95.159, -115.952],
        [106.066, -106.066],
        [115.952, -95.159],
        [124.72, -83.336],
        [132.288, -70.71],
        [138.582, -57.403],
        [143.541, -43.543],
        [147.118, -29.264],
        [149.278, -14.703],
        [150.0, -0.0],
        [149.278, 14.703],
        [147.118, 29.264],
        [143.541, 43.543],
        [138.582, 57.403],
        [132.288, 70.71],
        [124.72, 83.336],
        [115.952, 95.159],
        [106.066, 106.066],
        [95.159, 115.952],
        [83.336, 124.72],
        [70.71, 132.288],
        [57.403, 138.582],
        [43.543, 143.541],
        [29.264, 147.118],
        [14.703, 149.278],
        [0.0, 150.0],
        [-14.703, 149.278],
        [-29.264, 147.118],
        [-43.543, 143.541],
        [-57.403, 138.582],
        [-70.71, 132.288],
        [-83.336, 124.72],
        [-95.159, 115.952],
        [-106.066, 106.066],
        [-115.952, 95.159],
        [-124.72, 83.336],
        [-132.288, 70.71],
        [-138.582, 57.403],
        [-143.541, 43.543],
        [-147.118, 29.264],
        [-149.278, 14.703],
        [-150.0, 0.0],
        [-149.278, -14.703],
        [-147.118, -29.264],
        [-143.541, -43.543],
        [-138.582, -57.403],
        [-132.288, -70.71],
        [-124.72, -83.336],
        [-115.952, -95.159],
        [-106.066, -106.066],
        [-95.159, -115.952],
        [-83.336, -124.72],
        [-70.71, -132.288],
        [-57.403, -138.582],
        [-43.543, -143.541],
        [-29.264, -147.118],
        [-14.703, -149.278]
      ]
    }
  ];

  const joinArrays = (primary, defaults) => {
    for(let i=0; i<defaults.length; i++) {
      if(primary.find((a) => (a.name == defaults[i].name)) === undefined) {
        primary.push(defaults[i]);
      }
    }
    return primary
  };

  const [activeArrayIdx, setActiveArrayIdx] = React.useState(0);

  const [arrays, setArrays] = React.useState(
    joinArrays(
      JSON.parse(localStorage.getItem('mic_arrays')) || [],
      defaultArrays
    )
  );


  let activeArray = arrays[activeArrayIdx];

  // Publish the active array to parent one time a little while after mount
  // It seems the background worker needs some time to start up before it can receive messages
  React.useEffect(() => {
    setTimeout(() => {
      if(props.onMicChange) {
        console.log("Publishing mics");
        props.onMicChange(activeArray);
      }
    }, 500);
  }, []);

  React.useEffect(() => {
    updateScale(activeArray.positions, true);
  }, [])

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

  const handleDeleteArray = (array_idx) => {
    // Cowardly refuse to delete the last array. Sorry.
    if(arrays.length == 1) {
      console.log("Sorry Dave. I won't delete the last array because it will probably break my shitty code.");
      return;
    }
    let arraysCopy = [...arrays]
    arraysCopy.splice(array_idx, 1);

    if(array_idx >= arraysCopy.length) {
      setActiveArrayIdx(arraysCopy.length - 1);
    }
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

  const handleRestoreDefaults = () => {
    setArrays(defaultArrays);
    localStorage.setItem('mic_arrays', JSON.stringify(defaultArrays));
  }

  return (

    <Box sx={{ borderRadius: "10px", width: "fit-content", bgcolor: '#E7EBF0' }}>
      <div style={{ padding: "5px" }}>
        <span>To edit the microphone array: Drag dots around to re-arrange, Shift-click to add, Ctrl click to delete.</span>
        <Tooltip title="Deletes all array modifications from local storage!"><Button onClick={handleRestoreDefaults} variant="outlined" sx={{float: "right"}}>Restore Defaults</Button></Tooltip>
      </div>
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
              onDelete={handleDeleteArray}
              onSaveAll={handleSaveArrays}
              onImportJson={handleImportJson}

            />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
