
import React from 'react';
import PropTypes from 'prop-types';

import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import IconButton from '@mui/material/IconButton';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import ListSubheader from '@mui/material/ListSubheader';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';

import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ImportExportIcon from '@mui/icons-material/ImportExport';
import SaveIcon from '@mui/icons-material/Save';
import { Tune } from '@mui/icons-material';

export default function ArrayList(props) {
  let {arrays, selectedIdx} = props;

  let [newDialogOpen, setNewDialogOpen] = React.useState(false);
  let [newName, setNewName] = React.useState("");
  let [jsonDialogOpen, setJsonDialogOpen] = React.useState(false);
  let [jsonData, setJsonData] = React.useState("");
  let [jsonAlert, setJsonAlert] = React.useState(null);

  const handleAddArrayClick = (e) => {
    setNewDialogOpen(true);
  }

  const handleNewCancel = (e) => {
    setNewDialogOpen(false);
  }

  const handleNewSave = (e) => {
    console.log("Adding new array: ", newName);
    if(props.onAdd) {
      props.onAdd(newName)
    }
    setNewDialogOpen(false);
    setNewName("");
  }

  const handleNewNameChange = (e) => {
    setNewName(e.target.value);
  }

  const handleSave = (e) => {
    if(props.onSaveAll) {
      props.onSaveAll();
    }
  }

  const handleDelete = (i) => (e) => {
    e.stopPropagation();
    if(props.onDelete) {
      props.onDelete(i);
    }
  }

  const handleImportClick = () => {
    setJsonDialogOpen(true);
  }

  const handleJsonSave = () => {
    if(validateJson(jsonData)) {
      props.onImportJson(JSON.parse(jsonData));
      handleJsonClose();
    }

  }

  const handleJsonClose = () => {
    setJsonDialogOpen(false);
    setJsonAlert(null);
    setJsonData("");
  }

  const handleJsonValueChange = (e) => {
    let json = e.target.value;
    setJsonData(json);
    validateJson(json);
  }

  const validateJson = (text) => {
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      setJsonAlert("Cannot parse input as JSON");
      return false;
    }
    if(!Array.isArray(data)) {
      setJsonAlert("Top-level object must be an array");
      return false;
    }
    for(let i=0; i<data.length; i++) {
      if(!Array.isArray(data[i])) {
        setJsonAlert(`Each element in the top array must be an array of two numbers. Element ${i} is not an array`);
        return false;
      }
      if(data[i].length != 2) {
        setJsonAlert(`Each element in the top array must be an array of two numbers. Element ${i} is not length 2`);
        return false;
      }
      if(!Number.isFinite(data[i][0]) || !Number.isFinite(data[i][1])) {
        setJsonAlert(`Each element in the top array must be an array of two numbers. Element ${i} does not contain numbers`)
        return false;
      }
    }
    setJsonAlert(null);
    return true;
  }

  const newArrayDialog = (
    <Dialog open={newDialogOpen}>
      <TextField autoFocus margin="dense" id="name" label="Array Name" onChange={handleNewNameChange}/>
      <DialogActions>
        <Button onClick={handleNewCancel}>Cancel</Button>
        <Button onClick={handleNewSave}>Save</Button>
      </DialogActions>
    </Dialog>
  );

  const jsonDialog = (
    <Dialog open={jsonDialogOpen} onClose={handleJsonClose}>
      <DialogTitle>Import or Export JSON description of mic locations</DialogTitle>
      <DialogContent>
        { jsonAlert && <Alert severity="error">{jsonAlert}</Alert> }
        <Box
          width={400}
        >
          <TextField
            sx={{margin: 2}}
            multiline
            fullWidth
            label="JSON"
            maxRows={50}
            minRows={30}
            defaultValue={JSON.stringify(arrays[selectedIdx].positions)}
            onChange={handleJsonValueChange}
          />
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={handleJsonSave} disabled={jsonAlert || !jsonData}>Save</Button>
        <Button onClick={handleJsonClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  )


  return <div>

    {newArrayDialog}
    {jsonDialog}

    <ButtonGroup variant="text">
      <Tooltip title="Save current arrays to browser local storage">
        <Button onClick={handleSave} sx={{padding: 1}} startIcon={<SaveIcon />}>Save</Button>
      </Tooltip>
      <Tooltip title="Import/export positions as JSON">
        <Button onClick={handleImportClick} sx={{padding: 1}} startIcon={<ImportExportIcon />}>JSON</Button>
      </Tooltip>
      <Tooltip title="Create a new empty array">
        <Button onClick={handleAddArrayClick} sx={{padding: 1}} startIcon={<AddIcon />}>New</Button>
      </Tooltip>
    </ButtonGroup>

    <List
      subheader={
        <ListSubheader component="div" id="nested-list-subheader">
          Saved Arrays
        </ListSubheader>
      }
    >
      {
          arrays.map((a, i) => {
              return <ListItemButton
                  onClick={() => props.onSelect(i)}
                  selected={selectedIdx == i}
                  key={i}
                >
                  <ListItemText primary={a.name} />
                  <IconButton edge="end" aria-label="delete" onClick={handleDelete(i)}><DeleteIcon /></IconButton>
                </ListItemButton>
          })
      }
  </List>
  </div>
}

ArrayList.propTypes = {
  arrays: PropTypes.array.isRequired,
  selectedIdx: PropTypes.number,
  onSelect: PropTypes.func,
  onAdd: PropTypes.func,
  onDelete: PropTypes.func,
  onSaveAll: PropTypes.func,
  onImportJson: PropTypes.func,
}