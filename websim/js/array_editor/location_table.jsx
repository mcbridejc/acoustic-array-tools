import React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';

export default function LocationTable(props) {
  let {positions} = props;

  let [editing, setEditing] = React.useState({mic: null, coord: null, value: null});


  const handleValueChanged = (mic_idx, coord_idx) => (e) => {
    // Cache locally while editing; changes passed up when focus is lost
    // TODO: Some validation perhaps...
    setEditing({mic: mic_idx, coord: coord_idx, value: e.target.value});
  }

  const handleBlur = (mic_idx, coord_idx) => (e) => {
    props.onEdit(mic_idx, coord_idx, e.target.value)
    setEditing({mic: null, coord: null, value: null});
  }

  const handleRowMouseEnter = (i) => (e) => {
    if (props.onHighlight) {
      props.onHighlight(i);
    }
  }
  const handleRowMouseLeave = (e) => {
    if (props.onHighlight) {
      props.onHighlight(null);
    }
  }

  return (
    <TableContainer component={Paper}>
      <Table size="small" sx={{minWidth:200}} aria-label="Microphone positions">
        <TableHead>
          <TableRow>
            <TableCell>X (mm)</TableCell>
            <TableCell>Y (mm)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {
            positions.map((pos, idx) => {
              let x = pos[0];
              let y = pos[1];
              if (editing.mic == idx) {
                if (editing.coord == 0) {
                  x = editing.value;
                } else {
                  y = editing.value;
                }
              }
              return <TableRow margin="dense" key={idx} sx={{ background: props.highlight == idx ? "#eeeee4" : "#ffffff"  }} onMouseEnter={handleRowMouseEnter(idx)} onMouseLeave={handleRowMouseLeave}>
                <TableCell><TextField size="small" onBlur={handleBlur(idx, 0)} onChange={handleValueChanged(idx, 0)} id={`pos-x-${idx}`} variant="standard" value={x} /></TableCell>
                <TableCell><TextField size="small" onBlur={handleBlur(idx, 1)} onChange={handleValueChanged(idx, 1)} id={`pos-y-${idx}`} variant="standard" value={y} /></TableCell>
              </TableRow>
            })
          }
        </TableBody>
      </Table>
    </TableContainer>
  )
}