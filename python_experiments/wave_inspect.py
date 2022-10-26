#%%
%matplotlib widget
import wave
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
PREFIX = "capture"

filenames = [f"gui/{PREFIX}_{i}.wav" for i in range(6)]
audio = []
for fp in filenames:
    print(f"Reading {fp}")
    wav_samplerate, data = scipy.io.wavfile.read(fp)
    audio.append(data)
# %%
plt.figure()
plt.plot(audio[1][25000:35000])
plt.figure()
for i in range(10):
    start = i*10000
    subset = audio[1][start:start+1024]
    plt.psd(subset, Fs=wav_samplerate, NFFT=1024)
# %%
