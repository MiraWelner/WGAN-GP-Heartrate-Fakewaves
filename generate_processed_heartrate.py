"""
Mira Welner
June 2025

This script loads Dr Dey's heartrate data and reports the heartrate (60k/rr interval) as well as the QT intervals.
It then creates a training table of length=6000 for each snip for both RR and QT. It is recorded at 10Hz so this represents
600 seconds, or 10 minutes.
"""

import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import matplotlib.pyplot as plt

heartrate_hz = 10 #the x values are 10th of a second
ecg_hz = 500
snip_len = 500*10

def scale_data(x:np.ndarray) -> np.ndarray:
    x_scaled = 2*(x - x.min()) / (x.max() - x.min())-1
    return x_scaled

def process_qt(qt_distance_ms):
    """
    The QT x value is represented by the sum of the previous R-R distances
    rather than the timestamp.
    """
    scaled_qt = qt_distance_ms/350-1
    num_samples = len(scaled_qt)//snip_len
    scaled_heartrate_trimmed = scaled_qt[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    return heartrate_snips

def proccess_heartrate_file(rr_distance_ms,timestamps):
    bpm = 60000/rr_distance_ms
    f_interp_bpm = interp1d(timestamps, bpm, kind='linear')
    x_bpm = np.arange(timestamps.min(), timestamps.max()-1, 1/heartrate_hz)
    interpolated_bpm = f_interp_bpm(x_bpm)
    return interpolated_bpm

rr_18 = []
qt_18 = []
for i in range(1,5):
    file_path = f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    data_file = np.genfromtxt(file_path, skip_header=1, usecols=(1, 3), delimiter=',', filling_values=0.0, dtype=float)
    timestamps, rr_distance_ms = data_file.T
    new_rr = proccess_heartrate_file(rr_distance_ms,timestamps)
    rr_18.extend(new_rr)

rr_07 = []
qt_07 = []
for i in range(1,5):
    file_path = f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    data_file = np.genfromtxt(file_path, skip_header=1, usecols=(1, 3), delimiter=',', filling_values=0.0, dtype=float)
    timestamps, rr_distance_ms = data_file.T
    new_rr = proccess_heartrate_file(rr_distance_ms,timestamps)
    rr_07.extend(new_rr)


#the below files are very short so they need to be concatinated before being processed
short_files_rr = []
short_files_qt = []
short_files_timestamps = []
for i in range(1,28):
    file_path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    data_file = np.genfromtxt(file_path, skip_header=1, delimiter=',', filling_values=0.0, dtype=float)
    timestamps = data_file[:,0]
    if len(short_files_timestamps):
        timestamps+=short_files_timestamps[-1]
    rr_distance_s = data_file[:, 1].T
    short_files_rr.extend(rr_distance_s*1000)
    short_files_timestamps.extend(timestamps)

rr_11 = proccess_heartrate_file(np.array(short_files_rr),np.array(short_files_timestamps))

fig = plt.figure()
plt.plot(rr_11.flatten())
plt.savefig('heartrate_11.png')
plt.show()
plt.plot(rr_07)
plt.savefig('heartrate_07.png')
plt.show()
plt.plot(rr_18)
plt.savefig('heartrate_18.png')
plt.show()
np.savetxt("processed_data/heartrate_18.csv", np.vstack(rr_18), delimiter = ",")
np.savetxt("processed_data/heartrate_07.csv", np.vstack(rr_07), delimiter = ",")
np.savetxt("processed_data/heartrate_11.csv", rr_11, delimiter = ",")
