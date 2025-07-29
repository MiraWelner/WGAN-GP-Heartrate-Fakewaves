"""
Mira Welner
June 2025

This script loads Dr Dey's heartrate data and reports the heartrate (60k/rr interval) as well as the QT intervals.
It then creates a training table of length=6000 for each snip for both RR and QT. It is recorded at 10Hz so this represents
600 seconds, or 10 minutes.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

heartrate_hz = 10 #the x values are 10th of a second
ecg_hz = 500
snip_len = 500*10

def scale_data(x:np.ndarray) -> np.ndarray:
    x_scaled = 2*(x - x.min()) / (x.max() - x.min())-1
    return x_scaled

def process_heartrate_file(rr_distance_ms):
    rr_distance_ms = np.clip(rr_distance_ms, a_min=None, a_max=2000)
    x_axis = np.cumsum(rr_distance_ms)
    bpm = 60000/rr_distance_ms
    f_interp_bpm = interp1d(x_axis, bpm, kind='linear')
    x_bpm = np.arange(x_axis[0], x_axis[-1]-1, 1000/heartrate_hz)
    interpolated_bpm = f_interp_bpm(x_bpm)
    return interpolated_bpm, x_bpm

def process_patient_data(name:str, parts:int):
    bpm_data = []
    all_x_axis = np.array([])
    for i in range(1,parts+1):
        file_path = f"heartrate_data/{name}_24_hours_part{i}_v3_wholecaseRRiQTi.csv"
        data_file = np.genfromtxt(file_path, skip_header=1, usecols=([3]), delimiter=',', filling_values=0.0, dtype=float)
        rr_distance_ms = data_file.T
        new_bpm, x_axis = process_heartrate_file(rr_distance_ms)
        bpm_data.extend(new_bpm)
        if len(all_x_axis):
            prev_max = all_x_axis[-1]
            all_x_axis = np.append(all_x_axis, x_axis+prev_max)
        else:
            all_x_axis = np.append(all_x_axis, x_axis)
    data = np.vstack([np.array(bpm_data), all_x_axis])
    np.savetxt(f"processed_data/heartrate_{name}.csv", data, delimiter = ",")
    return data

patient_06 = process_patient_data('06-31-24', parts=2)
patient_09 = process_patient_data('09-40-14', parts=5)
patient_10 = process_patient_data('10-48-45', parts=4)
patient_11 = process_patient_data('11-03-38', parts=5)
patient_13 = process_patient_data('13-22-23', parts=4)
patient_14 = process_patient_data('14-17-50', parts=4)
