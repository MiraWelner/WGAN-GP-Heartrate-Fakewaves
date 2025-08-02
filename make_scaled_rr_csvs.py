"""
Mira Welner
June 2025

This script loads Dr Dey's heartrate data and sorts it. It then appends the raw rr data per each patient to
an array and prints that array to a csv corresponding to the patient.
"""

import numpy as np
from scipy.interpolate import interp1d
import glob
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
    all_rr = np.array([])
    for i in range(1,parts+1):
        for j in range(len(glob.glob(f"heartrate_data/{name}_24_hours_part{i}_v3_everyRRQTinputIntoEntropy_*.csv"))):
            f = glob.glob(f"heartrate_data/{name}_24_hours_part{i}_v3_everyRRQTinputIntoEntropy_Rel{j+1}_Abs*.csv")[0]
            data_file = np.genfromtxt(f, skip_header=1, usecols=([1]), delimiter=',')
            clipped_datafile = np.clip(data_file, a_min=0, a_max=1.25)
            all_rr = np.append(all_rr,clipped_datafile)

    np.savetxt(f"processed_data/heartrate_{name}.csv", scale_data(all_rr), delimiter = ",")

process_patient_data('06-31-24', parts=2)
process_patient_data('09-40-14', parts=5)
process_patient_data('10-48-45', parts=4)
process_patient_data('11-03-38', parts=5)
process_patient_data('13-22-23', parts=4)
process_patient_data('14-17-50', parts=4)
