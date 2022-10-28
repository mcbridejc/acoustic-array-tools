import React, {useState} from "react";
import ReactDOM from "react-dom";

import Container from '@mui/material/Container';
import Box from '@mui/material/Box';

import FreqSelect from "./freq_select.jsx";
import ArrayEditor from './array_editor/array_editor.jsx';
import PowerImage from '/power_image.jsx';


class App extends React.Component {

    constructor(props) {
        super(props)
        this.state = {
            power: Array(20*20).fill(0.0),
            sourceLoc: [0.0, 0.0],
        };
        this.handleMicChange = this.handleMicChange.bind(this);
        this.handleFreqChange = this.handleFreqChange.bind(this);
        this.handleSourceChange = this.handleSourceChange.bind(this);
        this.computeWorker = new Worker(new URL('./worker.js', import.meta.url));
    }

    componentDidMount() {
        const component = this;
        this.computeWorker.addEventListener('message', function(e) {
            const message = e.data;
            if(message.type == 'power') {
                component.setState({power: message.power});
            }
        });
    }

    handleFreqChange(value) {
        console.log("freq: ", value);
        this.computeWorker.postMessage({
            type: 'freq',
            start_freq: value[0],
            end_freq: value[1]
        });
    }

    handleMicChange(mics) {
        console.log("Updating mics");
        this.computeWorker.postMessage({
            type: 'mics',
            mic_positions: mics.positions,
        });
    }

    handleSourceChange(newLoc) {
        this.setState({sourceLoc: newLoc});
        this.computeWorker.postMessage({
            type: 'source',
            position: newLoc,
        });
    }

    render() {

        return (
            <Container>
                <Box display="flex" alignItems="center" justifyContent="center">
                    <Box margin={2}>
                        <h1>Acoustic Camera Simulation</h1>
                        <FreqSelect onChange={this.handleFreqChange} />
                        <PowerImage power={this.state.power} width={20} sourceLocation={this.state.sourceLoc} onSourceChange={this.handleSourceChange} />
                    </Box>
                </Box>
                <ArrayEditor onMicChange={this.handleMicChange} />
                <Box display="flex" justifyContent="center">
                    <Box margin={2}>
                    <h2>More info</h2>
                    <p>This app simulates an acoustic camera with a single white noise source. 
                        The location of the source can be adjusted anywhere on a 1.8x1.8 meter plane located 1m above the array.
                        To give you some point of reference, this yields a horizontal field-of-view of 42 degrees.
                    </p>
                    <p>
                        An example set of microphone arrays are pre-loaded, but you can edit these or add more by clicking, typing positions, or by importing a JSON array using the JSON button.
                        You can save these changes to local browser storage, so that they persist between sessions, but you have to click Save to do so. If you want to abandon your changes and go back to
                        the default arrays, the "Restore Defaults" button will clear your local storage.
                    </p>
                    <p>
                        The beamforming processing can operate on any frequency band, and this can be adjusting using the slider above the image. 
                    </p>
                    <p>
                        The image shows the total average power (normalized per fft bin, so it is independent of the selected frequency band) received at each focal point on a 20x20 grid. 
                        The power is adjusted so that it is plotted relative to the loudest pixel, set at 0dB. The range of the color scale can be adjusted using the vertical slider to the right of the image.
                    </p>


                    </Box>
                </Box>
            </Container>
        );
    }
}

export default App;

ReactDOM.render(<App />, document.getElementById("acoustic-root"));
