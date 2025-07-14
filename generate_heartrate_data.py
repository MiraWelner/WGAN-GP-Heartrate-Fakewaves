import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})


wave_0 = [np.sin(i/(1.5*np.pi))/2+1 for i in range(600)]
wave_1 = [np.sin(i/(8*np.pi))/2+1 for i in range(600)]
wave_2 = [np.sin(i/(16*np.pi))/2+1 for i in range(600)]
wave_3 = [np.sin(i/(32*np.pi))/2+1 for i in range(600)]

wave_4 = [np.sin(i/(40*np.pi))/2+1 for i in range(600)]
wave_5 = [np.sin(i/(4*np.pi))/2 + np.sin(i/(2*np.pi+2))/2 + 1
        for i in range(600) ]
wave_6 = [np.sin(i/(4*np.pi))/2**2+1 +np.sin(i/(16*np.pi+2))/2**2
        for i in range(600)]

wave_7 = [np.sin(i/(4*np.pi))/2**2+1 + np.sin(i/(2*np.pi+2))/2**2 + np.sin(i/(9*np.pi+2))/2**2
            for i in range(600)]



wave_8 = [np.sin(i/(4*np.pi))/2**2+1 +np.sin(i/(16*np.pi+2))/2**2-np.sin(i/(0.2*np.pi+2))/2**2
        for i in range(600)]
wave_9 = [np.sin(i/(40*np.pi))**2+0.5-np.sin(i/(0.2*np.pi+2))/2 for i in range(600)]

fig, axes = plt.subplots(10,2,figsize=(10,10))
for itr, wave in enumerate([wave_0, wave_1, wave_2, wave_3, wave_4, wave_5, wave_6, wave_7, wave_8,wave_9]):
    axes[itr,0].plot(wave)
    axes[itr,0].set_ylim(0,2)
    axes[itr,0].set_yticks([0,0.5,1,1.5,2])

    # Compute FFT and get magnitude spectrum
    n_half = len(wave) // 2
    fft_result = np.fft.fft(wave)
    freqs = np.fft.fftfreq(len(wave), d=1.0)
    posfreqs = freqs[:n_half-2]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:n_half-2]
    # Plot only positive frequencies (first half of spectrum)
    axes[itr,1].plot(posfreqs, posmag)
    axes[itr,1].set_xlim(0,0.1)
    axes[itr,1].set_ylim(0,150)
fig.tight_layout()
plt.savefig('figures/heartrate_data.png')
plt.show()
