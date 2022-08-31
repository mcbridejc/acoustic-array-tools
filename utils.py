import numpy as np
import struct
import scipy.io
import scipy.signal

def read_udp_capture(fname, num_channels):
    raw_bytes = [ [] for _ in range(num_channels)]
    with open(fname, 'rb') as f:
        while True:
            lenbytes = f.read(4)
            if len(lenbytes) < 4:
                break
            length = struct.unpack("<l", lenbytes)[0]
            packet = f.read(length)
            assert(len(packet) == length)
            assert(length == 482)
            pos = 0
            while pos <= length - 2 - num_channels:
                for ch in range(num_channels):
                    raw_bytes[ch].append(packet[pos + ch])
                pos += num_channels

    pdm = [np.unpackbits(np.array(rb, dtype=np.uint8), bitorder='little') for rb in raw_bytes]
    pdm = [p.astype(np.float32) - 0.5 for p in pdm]
    return pdm

def moving_avg_decimate(x, decimation, order):
    kernel = np.ones(decimation)
    for _ in range(order):
        x = scipy.signal.lfilter(kernel, [1], x)
    return x[0:-1:decimation]

# A lowpass with 8khz cutoff at 48k sample rate
final_filter = [ 
    1.01664764e-03,  2.33347679e-04, -4.15443380e-04, -8.91699472e-04,
    -5.73397878e-04,  4.25613754e-04,  1.16265899e-03,  7.23491164e-04,
    -7.13987999e-04, -1.78695457e-03, -1.19456151e-03,  8.13525761e-04,
    2.34254664e-03,  1.59237006e-03, -1.11615759e-03, -3.22795069e-03,
    -2.30046397e-03,  1.28287085e-03,  4.15321712e-03,  3.05168344e-03,
    -1.59354578e-03, -5.42614290e-03, -4.15254064e-03,  1.79902112e-03,
    6.86847371e-03,  5.44554341e-03, -2.10121647e-03, -8.75662216e-03,
    -7.21908091e-03,  2.31290902e-03,  1.10405170e-02,  9.46260437e-03,
    -2.58402158e-03, -1.40844365e-02, -1.25865409e-02,  2.76872575e-03,
    1.81546086e-02,  1.69755120e-02, -2.98203547e-03, -2.42107566e-02,
    -2.38673881e-02,  3.10972820e-03,  3.42681399e-02,  3.62207402e-02,
    -3.24501196e-03, -5.56600757e-02, -6.63897921e-02,  3.29239462e-03,
    1.39030399e-01,  2.73767217e-01,  3.29995956e-01,  2.73767217e-01,
    1.39030399e-01,  3.29239462e-03, -6.63897921e-02, -5.56600757e-02,
    -3.24501196e-03,  3.62207402e-02,  3.42681399e-02,  3.10972820e-03,
    -2.38673881e-02, -2.42107566e-02, -2.98203547e-03,  1.69755120e-02,
    1.81546086e-02,  2.76872575e-03, -1.25865409e-02, -1.40844365e-02,
    -2.58402158e-03,  9.46260437e-03,  1.10405170e-02,  2.31290902e-03,
    -7.21908091e-03, -8.75662216e-03, -2.10121647e-03,  5.44554341e-03,
    6.86847371e-03,  1.79902112e-03, -4.15254064e-03, -5.42614290e-03,
    -1.59354578e-03,  3.05168344e-03,  4.15321712e-03,  1.28287085e-03,
    -2.30046397e-03, -3.22795069e-03, -1.11615759e-03,  1.59237006e-03,
    2.34254664e-03,  8.13525761e-04, -1.19456151e-03, -1.78695457e-03,
    -7.13987999e-04,  7.23491164e-04,  1.16265899e-03,  4.25613754e-04,
    -5.73397878e-04, -8.91699472e-04, -4.15443380e-04,  2.33347679e-04,
    1.01664764e-03]

def pdm2pcm(x):

    x1 = moving_avg_decimate(x, 16, 4)
    x2 = scipy.signal.decimate(x1, 8, ftype='fir')
    #x3 = scipy.signal.lfilter(final_filter, [1], x2)
    x3 = x2
    x3 -= x3.mean()
    x3 *= 20 # Arbitrary gain
    return x3.astype(np.int16)