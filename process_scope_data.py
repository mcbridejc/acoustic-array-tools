# %%

%matplotlib widget
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.io
import struct


# %%

def edge_detect(x):
    threshold = x.mean()
    rising_msk = (x[:-1] < threshold) & (x[1:] > threshold)
    falling_msk = (x[1:] < threshold) & (x[:-1] > threshold)
    return np.argwhere(rising_msk), np.argwhere(falling_msk)

csv = pd.read_csv('scope_6.csv', skiprows=10)

clock = scipy.signal.medfilt(csv['CH2'], kernel_size=9)
data = scipy.signal.medfilt(csv['CH1'], kernel_size=9)


# %%
fig, axes = plt.subplots(2,1, sharex=True)

rising, falling = edge_detect(clock)
# Adjust sampling offset a bit before the clock
rising -= 10
falling -= 10

axes[0].plot(clock)
axes[1].plot(data)
axes[1].vlines(rising, ymin=-1, ymax=1, colors='green')
axes[1].vlines(falling, ymin=-1, ymax=1, colors='red')
axes[1].hlines([data.mean()], xmin=0, xmax=len(data))
# %%
mask = None
pattern = None
def knock_out_bytes(x, n):
    global mask
    global pattern
    pattern = np.ones(n*8, dtype=bool)
    pattern[-8:] = False
    mask = np.tile(pattern, int(len(x) / len(pattern) + 1))
    return x[mask[:len(x)]]

def knock_out_bits(x, n):
    global mask
    global pattern
    pattern = np.ones(n, dtype=bool)
    pattern[-1] = False
    mask = np.tile(pattern, int(len(x) / len(pattern) + 1))
    return x[mask[:len(x)]]

threshold = data.mean()
scope_ch0 = (data[rising] > threshold).astype(np.float32) - 0.5
scope_ch1 = (data[falling] < threshold).astype(np.float32) - 0.5
scope_ch1_knockout = knock_out_bits(scope_ch1, 100)
print(f"Length: {len(scope_ch0)}, {len(scope_ch1)}, {len(scope_ch1_knockout)}")
print(f"Means: {scope_ch0.mean()}, {scope_ch1.mean()}")

#%%
diff = scope_ch0[:-1] - scope_ch1[:1]
plt.figure()
plt.plot(diff[:100])
plt.figure()
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(scope_ch0[:, 0], drawstyle="steps", label='0')
axes[1].plot(scope_ch1, drawstyle="steps", label='1')

corr = np.correlate(scope_ch0, scope_ch1, mode='full')
#%%
# autocorr = np.correlate(ch1[:,0], ch1[:,0], mode='full')[len(ch0):]
# plt.figure()
# plt.plot(autocorr[:800])
# autocorr
print(len(scope_ch1))
print(len(scope_ch1_knockout))
print(pattern.shape)
a = np.array([1, 2, 3, 4])
# %%
plt.figure()
plt.plot(scope_ch0, drawstyle="steps" )
lpf = scipy.signal.decimate(scope_ch1_knockout[:,0], 128, ftype='fir') 
lpf = lpf - lpf.mean()
lpf *= 30
t_dec = np.arange(len(lpf)) * 128
#lpf = (np.convolve(ch0[:,0], np.ones(150)/150., mode='full') - 0.5) * 10 + 0.5
plt.plot(t_dec, lpf)
# %%
scipy.io.wavfile.write('shortsine.wav', int(3.05e6 / 128), lpf)
# %%

########## UDP data ############

INFILE = '8bittest.bin'
NUM_CHANNELS = 6

raw_bytes = [ [] for _ in range(NUM_CHANNELS)]
with open(INFILE, 'rb') as f:
    while True:
        lenbytes = f.read(4)
        if len(lenbytes) < 4:
            break
        length = struct.unpack("<l", lenbytes)[0]
        packet = f.read(length)
        assert(len(packet) == length)
        pos = 0
        while pos < length - 2 - NUM_CHANNELS:
            for ch in range(NUM_CHANNELS):
                raw_bytes[ch].append(packet[pos + ch])
            pos += NUM_CHANNELS


# %%
# %%
pdm = [np.unpackbits(np.array(rb[50000:50000+int(150000/8)], dtype=np.uint8), bitorder='little') for rb in raw_bytes]
pdm = [p.astype(np.float32) - 0.5 for p in pdm]

ch0 = scipy.signal.decimate(pdm[0], 128, ftype='fir') * 30
ch0 -= ch0.mean()
ch1 = scipy.signal.decimate(pdm[1], 128, ftype='fir') * 30
ch1 -= ch1.mean()
# %%

ch1_f = ch1 # scipy.signal.filtfilt(np.ones(4)/4., [1], ch1, method="pad")

t1 = np.arange(len(ch1_f))
t2 = np.arange(len(lpf))

plt.figure()
plt.plot(t1-7, ch1_f)
plt.plot(t2, lpf)

plt.figure()
plt.psd(ch1, NFFT=10000, marker='.', Fs=3.05e6/128)
plt.psd(lpf, NFFT=10000, marker=".", Fs=3.05e6/128)
# %%
autocorr = np.correlate(ch1, lpf, mode='full')[len(lpf):]
plt.figure()
plt.plot(autocorr)

# %%
