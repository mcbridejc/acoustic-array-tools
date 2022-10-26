# %%
# 

%matplotlib widget
from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.io
import scipy.signal
from tqdm import tqdm


def cicfilt(pdm, decimation, stages):
    comb = [0] * stages
    integral = [0] * stages

    decimated = []
    for i, xin in tqdm(enumerate(pdm)):
        x = xin
        for s in range(stages):
            integral[s] += x
            x = integral[s]
            
        if i % decimation == 0:
            for s in range(stages):
                y = x - comb[s]
                comb[s] = x
                x = y
            decimated.append(x)
            
    return decimated
            

        

    


# %%
class CICFilter(object):
    def __init__(self, decimation, stages):
        self.decimation = decimation
        self.stages = stages
        self.samples = [[0] * decimation for _ in range(stages)]
        self.last_delta = [0] * stages
        self.pos = 0
        
    def push_sample(self, sample):
        x = sample
        for i in range(self.stages):
            self.samples[i][self.pos] = x
            x = np.sum(self.samples[i])
        self.pos += 1
        if self.pos == self.decimation:
            for i in range(self.stages):
                y = x - self.last_delta[i]

            return y
        else:
            return None
        
    def filter(self, pdm):
        output = []
        for bit in tqdm(pdm):
            y = self.push_sample(bit)
            if y is not None:
                output.append(y)
        return output


INFILE = 'tone_deg.bin'
NUM_CHANNELS = 6

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
    x3 = scipy.signal.lfilter(final_filter, [1], x2)
    x3 -= x3.mean()
    x3 *= 20 # Arbitrary gain
    return x3.astype(np.int16)

#%%
pdm = read_udp_capture(INFILE, NUM_CHANNELS)
pcm = [pdm2pcm(p) for p in pdm]

SAMPLE_FREQ = int(3.072e6/128)

# %%
plt.figure()
for ch in range(NUM_CHANNELS):
    plt.psd(pcm[ch], NFFT=2000, Fs=SAMPLE_FREQ, label=f'ch{ch}', detrend='mean')
plt.title('PCM Power Spectra')
plt.legend()

plt.figure()
plt.plot(pcm[0])

# %%
scipy.io.wavfile.write('pcm0.wav', SAMPLE_FREQ, pcm[0])

# Acoular has TimeSamples class designed to read smaples from an h5 file. 
# We can keep them all in memory, but for now it's easier to just conform. 
# It looks like we could also create a `SamplesGenerater` and feed samples to
# acoular that way. 

all_samples = np.array([
    pcm[0], pcm[1], pcm[2], pcm[4], pcm[5]
], dtype=np.float32).T
import h5py
with h5py.File('tone_90deg.h5', 'w') as f:
    dataset = f.create_dataset('time_data', all_samples.shape, dtype='float32')
    dataset[:, :] = all_samples / 32768
    dataset.attrs['sample_freq'] = float(SAMPLE_FREQ)

# %%

import acoular
from micarray import array_x5

mic_grid = acoular.microphones.MicGeom()
mic_grid.mpos_tot = np.array(array_x5).T

ts = acoular.TimeSamples(name='tone_90deg.h5')
ps = acoular.PowerSpectra(time_data=ts, block_size=128, window='Hanning')
grid = acoular.RectGrid(x_min = -200, x_max= 200, y_min=-200, y_max=200, z=10, increment=10)
st = acoular.SteeringVector(grid=grid, mics=mic_grid)
bb = acoular.BeamformerBase(freq_data=ps, steer=st)
pm = bb.synthetic(500, 0)
Lm = acoular.L_p( pm )

# %%

plt.figure()
plt.imshow(Lm.T, origin='lower', extent=grid.extend(), interpolation='bicubic')
Lm

# %%
scipy.io.wavfile.write('test.wav', SAMPLE_FREQ, ts.data[:, 0].astype(np.int16))


# %%

moving_avg = scipy.signal.lfilter(np.ones(8), [1], pdm[2])*32
moving_avg_order2 = scipy.signal.lfilter(np.ones(8), [1], moving_avg)
moving_avg_order3 = scipy.signal.lfilter(np.ones(8), [1], moving_avg_order2)
moving_avg = moving_avg[0:-1:8]
moving_avg_order3 = moving_avg_order3[0:-1:8]
def count_bits(b):
    count = 0
    for bit in range(8):
        if b & (1<<bit):
            count += 1
    return count
# filter = CICFilter(8, 1)
# cic = filter.filter(pdm[2][:50000])
cic = cicfilt(pdm[2][:50000], 8, 3)
sum = [(count_bits(b) - 4) / 4 for b in raw_bytes[2][:6000]]

cic_slow =  scipy.signal.decimate(cic, 16, ftype='fir') * 0.3
moving_avg_slow = scipy.signal.decimate(moving_avg, 16, ftype='fir')
moving_avg_o3_slow = scipy.signal.decimate(moving_avg_order3, 16, ftype='fir')
just_scipy = scipy.signal.decimate(pdm[2], 128, ftype='fir') * 100

cic_slow = cic_slow / np.sqrt(np.mean(cic_slow**2))
moving_avg_slow = moving_avg_slow / np.sqrt(np.mean(moving_avg_slow**2))
moving_avg_o3_slow = moving_avg_o3_slow / np.sqrt(np.mean(moving_avg_o3_slow**2))
just_scipy = just_scipy / np.sqrt(np.mean(just_scipy**2))

cic_slow -= cic_slow.mean()
moving_avg_slow -= moving_avg_slow.mean()
moving_avg_o3_slow -= moving_avg_o3_slow.mean()
just_scipy -= just_scipy.mean()

t_pcm = np.arange(len(pcm)) * 16
plt.figure()
plt.plot(cic_slow, label='dec')
plt.plot(just_scipy, label='cic')
plt.legend()
#%%
plt.figure()
plt.psd(cic_slow, Fs=3.06e6/128, NFFT=2000, label="cic")
plt.psd(moving_avg_slow, Fs=3.06e6/128, NFFT=2000, label='moving avg')
plt.psd(moving_avg_o3_slow, Fs=3.06e6/128, NFFT=2000, label='O3 moving avg')
plt.psd(just_scipy, Fs=3.06e6/128, NFFT=2000, label='scipy decimate')
plt.legend()

sample_rate = int(3.06e6/128)
scipy.io.wavfile.write("cic.wav", sample_rate, cic_slow)
scipy.io.wavfile.write("moving_avg_slow.wav", sample_rate, moving_avg_slow)
scipy.io.wavfile.write("moving_avg_o3_slow.wav", sample_rate, moving_avg_o3_slow)
scipy.io.wavfile.write("just_scipy.wav", sample_rate, just_scipy)
#%%
plt.plot(pcm,)
# %%
autocorr = np.correlate(pdm[2], pdm[3], mode='full')[len(pdm[0])-2:]
plt.figure()
plt.plot(autocorr)
autocorr
#%%
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(pdm[2][:10000], drawstyle="steps")
axes[1].plot(pdm[3][1:10000], drawstyle="steps")
axes[2].plot(pdm[3][1:10000] - pdm[2][:10000-1], drawstyle="steps")
errors = pdm[3][:-1] - pdm[2][1:]
e_idx = np.argwhere(np.abs(errors) > 0)

#%%
epos = ((e_idx % (8*80)))
plt.figure()
plt.plot(errors,  drawstyle="steps")
# %%

ch0 = scipy.signal.decimate(pdm[2], 128, ftype='fir') * 30
ch0 -= ch0.mean()
ch1 = scipy.signal.decimate(pdm[3], 128, ftype='fir') * 30
ch1 -= ch1.mean()

plt.figure()
plt.psd(ch0)
plt.psd(ch1)

pdm_c = (pdm[0] + pdm[1]) / 2
ch_c = scipy.signal.decimate(pdm_c, 128, ftype='fir') * 30
ch_c -= ch_c.mean()

scipy.io.wavfile.write('udp_ch1.wav', int(3.07e6 / 128), ch1)
# %%
fig, axes = plt.subplots(1,1)
decimated_time = np.arange(len(ch0)) * 128
axes.plot(decimated_time[:1000], ch0[:1000])
#axes.plot(errors)
# %%
pdm[0].shape

# %%
raw_bytes[0].shape
# %%

ch0.shape
# %%
