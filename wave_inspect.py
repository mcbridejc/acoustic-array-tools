#%%
%matplotlib widget
import wave
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
PREFIX = "tone_0deg"

filenames = [f"{PREFIX}_{i}.wav" for i in range(6)]
audio = []
for fp in filenames:
    wav_samplerate, data = scipy.io.wavfile.read(fp)
    audio.append(data)
# %%
plt.figure()
plt.plot(audio[1][20000:25000])
plt.figure()
plt.psd(audio[1], Fs=wav_samplerate, NFFT=2048)
# %%
