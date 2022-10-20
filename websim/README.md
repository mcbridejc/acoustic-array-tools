# websim

A simulator for acoustic microphone arrays with React and WASM. 

The rust module in `src/lib.rs` is a wrapper around beamforming code in the `dsp` crate, which is
compiled to wasm to be used by the javascript app.

## Running

In the `js` directory, you can build the wasm module with `npm run build`. You can then launch a
development web server with `npm run dev`. 