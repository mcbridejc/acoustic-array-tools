# Acoustic Array Workspace

This is a collection of projects for acoustic phased array beamforming. It's very much a "working" repository with some bits more historical than useful, and supports ongoing development of acoustic arrays/cameras. Some more info on the project can be found [here](https://jeffmcbride.net/acoustic-beamforming/)

## Overview of the Repo

The `stm32` directory contains an embedded project targeting the [NUCLEO-H753ZI](https://www.st.com/en/evaluation-tools/nucleo-h753zi.html) dev board. It is setup to read audio from 6 PDM microphones wired to it, and process that audio to look for loud noises and determine the azimuth angle to them. When a noise is detected, a serial command is sent out USART1 (on PB6), which is intended for a BLDC servo controller, which can then point towards the noise. That's pretty much it. 

The `gui` directory contains a Rust GTK application which collects raw audio from UDP packets (sent by the Nucleo board, is the idea) and runs the same processing pipeline as the embedded application, displaying various charts and graphs of the data, including a waterfall diagram showing power vs azimuth over time and an image showing power on spatial grid (i.e. an acoustic camera). It's setup for six microphones, because that's what the NUCLEO based array has, but could readily be modified for more! 

The `websim` directory has a Rust and React project to run a simulation of an acoustic camera in the browser. The Rust is just a thin wrapper on the DSP library to compile it to a WASM interface, and the React app provides a UI for managing different microphone configurations, different acoustic source locations, etc, while displaying an image from the simulated array. It's intended for quickly experimenting with different acoustic array arrangements. There is a [live demo avialable here](https://jeffmcbride.net/acoustic-sim/).

The `dsp` directory has the core signal processing code, shared by all three of the above targets. 

The `clitools` is basically deprecated. At some point it should either get removed or refreshed. 

The `python_experiments` is more or less a scratchpad of tools that were useful during development and may become useful again. It can mostly be ignored.

## Building

All the rust code depends on nightly features, so you will need to install the nightly toolchain and enable it to build.

`rustup override set nightly`
