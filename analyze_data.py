import numpy as np
import os
import matplotlib.pyplot as plt


def get_samples(folder_name):
    for file in os.listdir(f'processed_data/{folder_name}/'):
        data = np.loadtxt(f'processed_data/{folder_name}/{file}', delimiter=',')
        for i in range(5):
            _ = plt.figure(figsize=(10,5))
            plt.title(f"Patient {file.split("_")[0]} From the {folder_name} Group Sample")
            plt.plot(data[i,:])
            plt.xticks(np.arange(0,4000, 500), [str(i) for i in np.arange(0,8,1)])
            plt.ylabel("Volts")
            plt.xlabel("Seconds")
            plt.savefig(f'figures/example_snips/{folder_name}_{file.split("_")[0]}_{i}.png')
            plt.close()
#get_samples('alive')
#get_samples('dead')
#get_samples('ten_patients')


def get_averages(folder_name):
    for file in os.listdir(f'processed_data/{folder_name}/'):
        data = np.loadtxt(f'processed_data/{folder_name}/{file}', delimiter=',')
        _ = plt.figure(figsize=(10,5))

        signal = np.mean(data, axis=0)
        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)
        error = np.array([signal - lower, upper - signal])


        plt.plot(signal, color='cornflowerblue')
        plt.errorbar(range(0, len(signal)), signal, yerr=error,alpha=0.1, color='cornflowerblue')

        plt.title(f"Patient {file.split("_")[0]} From the {folder_name} Group Mean and 95% Confidence")
        plt.xticks(np.arange(0,4000, 500), [str(i) for i in np.arange(0,8,1)])
        plt.ylabel("Volts")
        plt.xlabel("Seconds")
        plt.savefig(f'figures/averages_confidences/{folder_name}_{file.split("_")[0]}.png')
        plt.close()
'''
get_averages('dead')
get_averages('ten_patients')
get_averages('alive')
'''

def get_fft(folder_name):
    for file in os.listdir(f'processed_data/{folder_name}/'):
        data = np.loadtxt(f'processed_data/{folder_name}/{file}', delimiter=',')
        n = data.shape[1]
        freqs = np.fft.fftfreq(n, d=1/500)
        for i in range(5):
            _ = plt.figure(figsize=(10,5))
            plt.title(f"Patient {file.split("_")[0]} From the {folder_name} Group Sample FFT")
            signal = data[i, :]
            fft_vals = np.fft.fft(signal)
            fft_magnitude = np.abs(fft_vals)

            plt.plot(freqs, fft_magnitude)
            plt.ylabel("Magnitude")
            plt.xlabel("Frequency")
            plt.savefig(f'figures/ffts/{folder_name}_{file.split("_")[0]}_{i}.png')
            plt.close()
"""
get_fft('alive')
get_fft('ten_patients')
get_fft('dead')
"""
