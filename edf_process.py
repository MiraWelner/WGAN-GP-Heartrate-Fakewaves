

import pyedflib
import matplotlib.pyplot as plt
import numpy as np


def load_edf(edf_name, plot=False):
    edf_file_path = f'heartrate_data/{edf_name}.EDF'
    edf = pyedflib.EdfReader(edf_file_path)
    n_signals = edf.signals_in_file
    signal_labels = edf.getSignalLabels()
    signals = []
    sampling_rates = []
    durations = []
    for i in range(n_signals):
        sig = edf.readSignal(i)
        fs = edf.getSampleFrequency(i)
        duration = len(sig) / fs
        signals.append(sig)
        sampling_rates.append(fs)
        durations.append(duration)

    edf._close()
    if plot:
        fig1, axs1 = plt.subplots(n_signals, 1, figsize=(12, 2 * n_signals), sharex=True,  layout = "constrained")
        for i in range(n_signals):
            time = np.linspace(0, durations[i], len(signals[i]))
            ax = axs1[i] if n_signals > 1 else axs1
            ax.plot(time, signals[i])
            ax.set_title(f"{signal_labels[i]} (fs = {sampling_rates[i]} Hz)")
            ax.set_ylabel("Amplitude")
        axs1[-1].set_xlabel("Time (s)")
        fig1.suptitle("All Channels from EDF file", fontsize=16)
        plt.show()

load_edf('06-31-24')
load_edf('10-48-45')
load_edf('11-03-38')
load_edf('09-40-14')

load_edf('13-22-23')
load_edf('14-17-50')
