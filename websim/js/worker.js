
const wasm = import("../build/beamformjs");

wasm.then(({System}) => {
  console.log("Starting worker");

  const beamformer = System.new();

  mics = null;
  micsDirty = false;

  source = [0.0, 0.0, 1.0];

  start_freq = 2000.0;
  end_freq = 3000.0;

  self.addEventListener(
    "message",
    function(e) {
      let msg = e.data;
      if (msg.type == 'mics') {
        mics = msg.mic_positions;
        micsDirty = true;
      } else if (msg.type == 'freq') {
        start_freq = msg.start_freq;
        end_freq = msg.end_freq;
        console.log("Freq: ", start_freq, end_freq);
      } else if(msg.type == 'source') {
        source[0] = msg.position[0];
        source[1] = msg.position[1];
        console.log("New source location: ", msg.position);
      }
    },
    false
  );

  self.setInterval(() => {
    if (mics) {
      if (micsDirty && mics.length > 0) {
        console.log("Updating mic array");
        beamformer.set_mics(mics.flat());
        micsDirty = false;
      }

      console.log("Starting beamformer");
      let power = beamformer.run(source[0], source[1], source[2], start_freq, end_freq);
      console.log("Beamformer finished");
      self.postMessage({
        type: 'power',
        power: power,
      });
    } else {
    }
  }, 500);
});
