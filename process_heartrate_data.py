"""
Mira Welner
June 2025

This script loads Dr. Cong's PPG and ECG data, removes outliers, and interpolates the PPG data
which is 125 hz to be 500hz like the ECG data.

As of now, this algorithm does NOT remove noise because my previous notch filter method
cannot easily be automated. So, my hope is that the model will remove the noise.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from glob import glob

frac_sec = 10 #the x values are 100th of a second

def process_rr(rr_distance_ms, snip_len = 3500):
    bpm = 60000/rr_distance_ms
    scaled_heartrate = bpm/75 -1
    num_samples = len(scaled_heartrate)//snip_len
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df

def process_qt(qt_distance_ms, snip_len = 3500):
    scaled_qt = qt_distance_ms/350-1
    num_samples = len(scaled_qt)//snip_len
    scaled_heartrate_trimmed = scaled_qt[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df

def proccess_file(path):
    file = pd.read_csv(path)
    rr_distance_ms = file.iloc[:, 3]
    qt_distance_ms = pd.Series(pd.to_numeric(file.iloc[:, 4], errors='coerce')).fillna(0)
    timestamps = file.iloc[:,1]
    qt_times_s = [sum(rr_distance_ms[:i])/1000 for i in range(len(rr_distance_ms))]
    f_interp_rr = interp1d(timestamps, rr_distance_ms, kind='linear')
    x_rr = np.arange(min(timestamps), max(timestamps)-1, 1/frac_sec)
    interpolated_rr = f_interp_rr(x_rr)

    f_interp_qt = interp1d(qt_times_s, qt_distance_ms, kind='linear')
    x_qt = np.arange(min(qt_times_s), max(qt_times_s)-1, 1/frac_sec)
    interpolated_qt = f_interp_qt(x_qt)
    return process_rr(interpolated_rr), process_qt(interpolated_qt)



rr = pd.DataFrame()
qt = pd.DataFrame()

for i in range(1,5):
    path = f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    new_rr, new_qt = proccess_file(path)
    rr = pd.concat([rr, new_rr])
    qt = pd.concat([qt, new_qt])

for i in range(1,5):
    path = f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    new_rr, new_qt = proccess_file(path)
    rr = pd.concat([rr, new_rr])
    qt = pd.concat([qt, new_qt])


#the below files are very short so they need to be concatinated before being processed

short_files_rr = []
short_files_qt = []
short_files_timestamps = []
for i in range(1,28):
    path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    file = pd.read_csv(path)
    timestamps = file.iloc[:,0]
    if len(short_files_timestamps):
        timestamps+=short_files_timestamps[-1]
    rr_distance_s = np.asarray(file.iloc[:,1])
    qt_distance_s = np.asarray(file.iloc[:,2])
    short_files_rr.extend(rr_distance_s*1000)
    short_files_qt.extend(qt_distance_s*1000)
    short_files_timestamps.extend(timestamps)

qt_times_s = [sum(short_files_rr[:i])/1000 for i in range(len(short_files_rr))]
f_interp_rr = interp1d(short_files_timestamps, short_files_rr, kind='linear')
x_rr = np.arange(min(short_files_timestamps), max(short_files_timestamps)-1, 1/frac_sec)
interpolated_rr = f_interp_rr(x_rr)

f_interp_qt = interp1d(qt_times_s, short_files_qt, kind='linear')
x_qt = np.arange(min(qt_times_s), max(qt_times_s)-1, 1/frac_sec)
interpolated_qt = f_interp_qt(x_qt)
rr = pd.concat([rr, process_rr(interpolated_rr)])
qt = pd.concat([qt, process_qt(interpolated_qt)])

rr.to_csv('processed_data/rr_processed.csv')
qt.to_csv('processed_data/qt_processed.csv')
