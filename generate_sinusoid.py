"""
Mira Welner
July 2025
This script generates 10 different sinusouidal functions of varying complexities which can be used
to train and evaluate a GAN before it is used to analyze heartrate data.
It creates a plot (sinusoid.png) in the figures folder and a datafile (sinusoid.csv) in the
processed_data folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 7})

period = np.linspace(0, 4 * np.pi, 600)

wave_0 = [np.sin(10*i) for i in period]
wave_1 = [np.sin(i/2)*0.7 + np.sin(i)*0.2 -np.sin(i*20)*0.1 for i in period]
wave_2 = [np.sin(i)*0.8+np.sin(i*5)*0.2 for i in period]
wave_3 = [np.sin(i/2) for i in period]
wave_4 = [np.sin(6*i)*0.6 + np.sin(5*i)*0.4 for i in period]
wave_5 = [np.sin(i*5)*0.50 + np.sin(i*8)*0.25 + np.sin(i)*0.25 for i in period]
wave_6 = [np.sin(i)*0.5-np.sin(20* i)*0.5 for i in period]
wave_7 = [np.sin(i*5)/4 + np.sin(i*3)/4+ np.sin(i*4)/4 - np.sin(i*40)/4 for i in period]
wave_8 = [np.sin(i*2)*0.3 +np.sin(i+np.pi)*0.4-np.sin(i)*0.3 for i in period]
wave_9 = [np.sin(i*2)*0.1 +
          np.sin(i*3)*0.2 +
          np.sin(i*6)*0.3 +
          np.sin(i*8)*0.3 -
          np.sin(i*40)*0.1
          for i in period]

sin_heartrate_sim = np.array([wave_0, wave_1, wave_2, wave_3, wave_4, wave_5, wave_6, wave_7, wave_8,wave_9])
fig, axes = plt.subplots(5,4,figsize=(17,9))
for itr, wave in enumerate(sin_heartrate_sim):
    axes[itr//2,itr%2*2].plot(wave)
    axes[itr//2,itr%2*2].set_ylim(-1.1,1.1)
    axes[itr//2,itr%2*2].set_yticks([-1,0,1])
    axes[itr//2,itr%2*2].set_title(f'mean:{np.mean(wave):.3f}, std:{np.std(wave):.2f}')
    axes[itr//2,itr%2*2].set_ylabel(f"Sinusoid {itr}")
    # Compute FFT and get magnitude spectrum
    n_half = len(wave) // 2
    normalized_wave = wave-np.mean(wave)
    fft_result = np.fft.fft(normalized_wave)
    freqs = np.fft.fftfreq(len(normalized_wave), d=1.0)
    posfreqs = freqs[:n_half-2]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:n_half-2]
    # Plot only positive frequencies (first half of spectrum)
    axes[itr//2,itr%2*2+1].plot(posfreqs, posmag)
    peaks = posfreqs[find_peaks(posmag, height=10)[0]]
    peaks_list = ', '.join(f"{num:.3f}" for num in peaks)
    axes[itr//2,itr%2*2+1].set_title(f'Peak frequencies: {peaks_list}')
   # axes[itr//2,itr%2*2+1].set_xlim(0,0.2)
fig.tight_layout()
np.savetxt("processed_data/sinusoid.csv", sin_heartrate_sim, delimiter=",")
plt.savefig('figures/sinusoid.png')
plt.show()
