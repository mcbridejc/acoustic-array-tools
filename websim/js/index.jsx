import React, {useState} from "react";
import ReactDOM from "react-dom";

import Container from '@mui/material/Container';
import Box from '@mui/material/Box';

import FreqSelect from "./freq_select.jsx";
import ArrayEditor from './array_editor/array_editor.jsx';
import PowerImage from '/power_image.jsx';

const wasm = import("../build/beamformjs");



class App extends React.Component {

    constructor(props) {
        super(props)
        this.state = {
            power: Array(20*20).fill(0.0),
            sourceLoc: [0.5, 0.5],
        };
        this.handleMicChange = this.handleMicChange.bind(this);
        this.handleFreqChange = this.handleFreqChange.bind(this);
        this.handleSourceChange = this.handleSourceChange.bind(this);
        this.computeWorker = new Worker(new URL('./worker.js', import.meta.url));
        
    }

    componentDidMount() {
        let {system} = this.props;
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
        const {system} = this.props;

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
            </Container>
        );
    }
}

wasm.then(({System}) => {
    const system = System.new();

    console.log(system)
    
    ReactDOM.render(<App system={system} />, document.getElementById("root"));
});
