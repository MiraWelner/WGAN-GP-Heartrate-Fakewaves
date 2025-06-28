"""
Mira Welner
June 2025

This script loads Dr. Cong's PPG and ECG data, removes outliers, and interpolates the PPG data
which is 125 hz to be 500hz like the ECG data.

As of now, this algorithm does NOT remove noise because my previous notch filter method
cannot easily be automated. So, my hope is that the model will remove the noise.
"""

import numpy as np
import os
from scipy.signal import butter, sosfilt, find_peaks
from tqdm import tqdm

seconds = 7
hz = 500

'''
Design the band pass filter in this paper
https://ieeexplore.ieee.org/abstract/document/7726845
'''
lowcut = 0.5
highcut = 40.0
order = 4
nyq = 0.5 * hz
low = lowcut / nyq
high = highcut / nyq
sos = butter(order, [low, high], btype='bandpass', output='sos')


"""
Reads the signal from the file, runs it through the bandpass filter,
clips it between ranges for which less than 5% is in the histogram bin
normalizes it between -1 and 1
"""
def make_ecg_signal(path):
    signal = np.nan_to_num(np.fromfile(path, dtype=np.float64), nan=0.0)
    signal = sosfilt(sos, signal)
    counts, bin_edges = np.histogram(signal, bins=500)
    bin_edge_spots = np.where(counts > len(signal)//30)[0]
    bin_edge_spots = bin_edge_spots[bin_edge_spots != 0]  # it should never be 0 that means its getting no signal
    min_val = bin_edges[bin_edge_spots[0]]
    max_val = bin_edges[bin_edge_spots[-1]+1]
    signal = np.array(signal)[(signal<max_val) & (signal>min_val)]
    normalized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1
    return normalized_signal

def proccess_patient_files(input_folder, output_folder):
    for p in tqdm(os.listdir(input_folder)):
        segment_length = 500 * seconds
        path = f'{input_folder}/{p}'
        signal = make_ecg_signal(path)
        peaks = find_peaks(signal, height=0.8, distance=300)[0] #distance: 100bpm
        valid_peaks = peaks[peaks + segment_length <= len(signal)]
        segments = np.array([signal[peak:peak + segment_length] for peak in valid_peaks])
        np.savetxt(f'{output_folder}/{p}.csv', segments, delimiter=',', fmt='%.6f')

proccess_patient_files('DeadvsAlive/Dead', '../processed_data/dead')
proccess_patient_files('Additional10', '../processed_data/ten_patients')
proccess_patient_files('DeadvsAlive/Alive', '../processed_data/alive')
