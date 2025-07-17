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

wave_0 = [np.sin(i/(1.5*np.pi)) for i in range(600)]
wave_1 = [np.sin(i/(8*np.pi))*0.7 + np.cos(i)*0.3 for i in range(600)]
wave_2 = [np.sin(i/(16*np.pi))*0.8 + np.cos(i/8)*0.2 for i in range(600)]
wave_3 = [np.sin(i/(32*np.pi)) for i in range(600)]
wave_4 = [np.sin(i/(32*np.pi))*0.6+np.cos(i/(32*np.pi))*0.4 for i in range(600)]
wave_5 = [np.sin(i/(4*np.pi))*0.4 + np.sin(i/(2*np.pi+2))*0.6 for i in range(600) ]
wave_6 = [np.sin(i/(1.5*np.pi))/2 - np.cos(i*100)/2 for i in range(600)]
wave_7 = [np.sin(i/(4*np.pi)-0.5*np.pi)/3 + np.sin(i/(2*np.pi+2))/3+ np.sin(i/(9*np.pi+2))/3 for i in range(600)]
wave_8 = [np.sin(i/(4*np.pi))/3 +np.sin(i/(16*np.pi+2))/3-np.sin(i/(0.2*np.pi+2))/3 for i in range(600)]
wave_9 = [np.cos(i/(40*np.pi))/2-np.sin(i/(0.2*np.pi+2))/2 for i in range(600)]

sin_heartrate_sim = np.array([wave_0, wave_1, wave_2, wave_3, wave_4, wave_5, wave_6, wave_7, wave_8,wave_9])
fig, axes = plt.subplots(5,4,figsize=(17,9))
for itr, wave in enumerate(sin_heartrate_sim):
    axes[itr//2,itr%2*2].plot(wave)
    axes[itr//2,itr%2*2].set_ylim(-1.1,1.1)
    axes[itr//2,itr%2*2].set_yticks([-1,0,1])
    axes[itr//2,itr%2*2].set_title(f'mean:{np.mean(wave):.2f}, std:{np.std(wave):.2f}')
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
    peaks = posfreqs[find_peaks(posmag)[0]]
    peaks_list = ', '.join(f"{num:.3f}" for num in peaks)
    axes[itr//2,itr%2*2+1].set_title(f'Peak frequencies: {peaks_list}')
    axes[itr//2,itr%2*2+1].set_xlim(0,0.2)
fig.tight_layout()
np.savetxt("processed_data/sinusoid.csv", sin_heartrate_sim, delimiter=",")
plt.savefig('figures/sinusoid.png')
plt.show()
