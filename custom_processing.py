#%%
%matplotlib ipympl
import matplotlib.pyplot as plt
from utils import read_udp_capture, pdm2pcm
from micarray import array_x5 
import math
import numpy as np
from scipy.spatial.distance import cdist
import scipy.fft

SPEED_OF_SOUND = 343.0 # m/s
NFFT = 1024
SAMPLE_RATE = 24000
NUM_CHANNELS = 6


# %%
audio = np.array([pdm2pcm(p) for p in read_udp_capture('tone_0deg.bin', 6)])
audio = audio[[1,0,2,5,4]]

#%%
import socket


def read_udp_packets(n):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_rx:
        udp_rx.bind(("0.0.0.0", 10200))
        packets = []
        for _ in range(n):
            data, _addr = udp_rx.recvfrom(512)
            packets.append(data)
        #udp_rx.close()
    return packets

#%%

def get_focal_points(n, radius=1.0, z=0.1): 
    points = np.zeros((n, 3))
    alpha = np.arange(0, math.pi * 2, math.pi*2 / n)
    points[:, 0] = np.sin(alpha) * radius
    points[:, 1] = np.cos(alpha) * radius
    points[:, 2] = z
    return alpha, points

class BeamForm():
    def __init__(self, steering_vectors, nfft):
        self.sv = steering_vectors
        self.nfft = nfft
        #self.fft_freqs = scipy.fft.fftfreq(NFFT)[:int(NFFT/2)] * SAMPLE_RATE
    
    def get_powers(self, samples):
        # Compute the FFT of each audio channel
        ffts = np.array([scipy.fft.fft(ch, n=NFFT)[0:int(NFFT/2)] for ch in samples])
        # Compute the cumulative FFT after adjusting phases
        summed = np.array([(ffts * sv).sum(axis=0) for sv in self.sv])  / ffts.shape[0]
        powers = (np.abs(summed)[:, 15:200]**2).mean(axis=1)
        return powers


# Setup the "focal points". These are points in space at which we will focus
# the array to measure power. They are setup as a ring around the array, because
# we're really only trying to find an azimuth angle.
focal_angles, focal_points = get_focal_points(100, 0.3, 0.1)
# Get the microphone positions
mics = np.array(array_x5)

# create a mics x focal_points matrix with distance between each pairing
distance_matrix = cdist(focal_points, mics)

def steering_vector(distance_matrix, frequencies):
    sv = np.zeros(distance_matrix.shape + (len(frequencies),), dtype=np.complex64)
    for i, f in enumerate(frequencies):
        if f > 0:
            sv[:, :, i] = np.exp(1j * math.pi * 2 * f * distance_matrix / SPEED_OF_SOUND)
        else:
            sv[:, :, i] = np.exp(0j)
    return sv
    
# Get a "steering vector", which has phase adjustments for each frequency,
# for each focal point, at each microphone
fft_freqs = scipy.fft.fftfreq(NFFT)[:int(NFFT/2)] * SAMPLE_RATE
sv = steering_vector(distance_matrix, fft_freqs)

beamform = BeamForm(sv, NFFT)
line = None

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)  # theta increasing clockwise
ax.set_title('Where are you???')
ax.grid(False)
arrow = plt.arrow(0, 0.5, 0, 1, alpha = 0.5, width = 0.015,
                 edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)


# %%
while True:
    packets = read_udp_packets(250)
    # validate packets
    seq_counter = [p[480] for p in packets]
    diff = np.diff(seq_counter)
    errors = (diff!=1).sum()
    if errors > 1:
        print(f"Errors: {errors}")
    channels = [ b'' for _ in range(5)]
    for p in packets:
        # Skip ch4 -- the mic is missing
        for i, ch in enumerate([1,0,2,5,4]):
            # Last two bytes of each packet are metadata we can ignore
            channels[i] += p[ch:-2:6]

    pdm = [np.unpackbits(np.array([b for b in bs], dtype=np.uint8), bitorder='little') for bs in channels]
    pdm = [p.astype(np.float32) - 0.5 for p in pdm]

    pcm = np.array([pdm2pcm(p) for p in pdm])
    pcm = pcm[:, 10:-10]

    powers = beamform.get_powers(pcm)

    total_power = powers.mean()
    
    # if total_power < 5e6:
    #     powers = 0.5 * np.ones_like(powers)
    # else:
    #     powers = (powers - powers.min()) / (powers.max() - powers.min())
    if line is None:
        line = ax.plot(focal_angles, powers)[0]
    else:
        line.set_ydata(powers)
    peak_idx = np.argmax(powers)
    peak_angle = focal_angles[peak_idx]
    #print(peak_angle)
    arrow.set_data(x=peak_angle, y=powers.min(), dx=0.0, dy=powers.max() - powers.min())
    ax.set_ylim(powers.min(), powers.max())
    fig.canvas.draw()

# %%

# a is about 116mm further from source than b. 
# This should result in 0.338ms delay.
# At the 584Hz freq, that's a phase shift of 0.197
a = pcm[2]
b = pcm[4]

plt.figure(); plt.plot(a, label='a'); plt.plot(b, label='b')
plt.legend()

ffta = scipy.fft.fft(a, n=NFFT)[0:int(NFFT/2)]
fftb = scipy.fft.fft(b, n=NFFT)[0:int(NFFT/2)]

plt.figure()
plt.psd(a, NFFT=1024, Fs=SAMPLE_RATE)
plt.psd(b, NFFT=1024, Fs=SAMPLE_RATE)

phase_a = np.zeros_like(ffta)
for i, f in enumerate(fft_freqs):
    if f > 0:
        wavelength = SPEED_OF_SOUND / f
        phase_a[i] = np.exp(1j * math.pi * 2 * f * 0.10 / SPEED_OF_SOUND)

ffta_delay = ffta * phase_a

combo1 = (ffta + fftb) / 2
combo2 = (ffta_delay + fftb) / 2

plt.figure()
plt.plot(np.abs(combo1), label='orig')
plt.plot(np.abs(combo2), label='delayed')
plt.yscale('log')
plt.legend()

plt.figure();
plt.plot(np.angle(ffta_delay), label='adelay')
plt.plot(np.angle(ffta), label='a')
plt.plot(np.angle(fftb), label='b')
plt.legend()

for 
# %%

powers = beamform.get_powers(pcm)
plt.figure(); plt.plot(focal_angles*180/math.pi, powers, '.')


# %%
|