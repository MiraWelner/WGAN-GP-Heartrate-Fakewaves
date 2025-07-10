"""
Mira Welner
June 2025

This script loads Dr Dey's ECG data and reports the RR and QT intervals. It then creates a training table of
length=3500 for each snip for both RR and QT
"""

import polars as pl
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import itertools

frac_sec = 10 #the x values are 10th of a second

def process_rr(rr_distance_ms, snip_len):
    bpm = 60000/rr_distance_ms
    scaled_heartrate = bpm/75 -1
    num_samples = len(scaled_heartrate)//snip_len
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pl.DataFrame(heartrate_snips)
    return snip_df

def process_qt(qt_distance_ms, snip_len):
    scaled_qt = qt_distance_ms/350-1
    num_samples = len(scaled_qt)//snip_len
    scaled_heartrate_trimmed = scaled_qt[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pl.DataFrame(heartrate_snips)
    return snip_df

def proccess_file(path, snip_len=3500):
    file = pl.read_csv(path)
    rr_distance_ms = np.array(file.select(file.columns[3]), dtype=np.float64).flatten()
    qt_distance_ms = [i.astype(np.float64).item() if i != '  ' else 0 for i in np.array(file.select(file.columns[4]))]
    timestamps = np.array(file.select(file.columns[1]), dtype=np.float64).flatten()
    qt_times_s = np.array([s / 1000 for s in itertools.accumulate(rr_distance_ms)], dtype=np.float64).flatten()
    f_interp_rr = interp1d(timestamps, rr_distance_ms, kind='linear')
    x_rr = np.arange(timestamps.min(), timestamps.max()-1, 1/frac_sec)
    interpolated_rr = f_interp_rr(x_rr)

    f_interp_qt = interp1d(qt_times_s, qt_distance_ms, kind='linear')
    x_qt = np.arange(qt_times_s.min(), qt_times_s.max()-1, 1/frac_sec)
    interpolated_qt = f_interp_qt(x_qt)
    return process_rr(interpolated_rr, snip_len), process_qt(interpolated_qt, snip_len)


#for the large processed files which can be hours long
rr = pl.DataFrame()
qt = pl.DataFrame()

#for the 10 minute ones
rr_18 = pl.DataFrame()
qt_18 = pl.DataFrame()

rr_07 = pl.DataFrame()
qt_07 = pl.DataFrame()

rr_11 = pl.DataFrame()
qt_11 = pl.DataFrame()

for i in range(1,5):
    path = f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    new_rr, new_qt = proccess_file(path)
    rr = pl.concat([rr, new_rr])
    qt = pl.concat([qt, new_qt])

    rr_10min, qt_10min = proccess_file(path, snip_len=6000)
    rr_18 = pl.concat([rr_18, rr_10min])
    qt_18 = pl.concat([qt_18, qt_10min])

for i in range(1,5):
    path = f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    new_rr, new_qt = proccess_file(path)
    rr = pl.concat([rr, new_rr])
    qt = pl.concat([qt, new_qt])

    rr_10min, qt_10min = proccess_file(path, snip_len=6000)
    rr_07 = pl.concat([rr_07, rr_10min])
    qt_07 = pl.concat([qt_07, qt_10min])

#the below files are very short so they need to be concatinated before being processed

short_files_rr = []
short_files_qt = []
short_files_timestamps = []
for i in range(1,28):
    path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    file = pl.read_csv(path)
    timestamps = np.array(file.select(file.columns[0]), dtype=np.float64).flatten()
    if len(short_files_timestamps):
        timestamps+=short_files_timestamps[-1]
    rr_distance_s = np.array(file.select(file.columns[1]), dtype=np.float64).flatten()
    qt_distance_s = np.array(file.select(file.columns[2]), dtype=np.float64).flatten()
    short_files_rr.extend(rr_distance_s*1000)
    short_files_qt.extend(qt_distance_s*1000)
    short_files_timestamps.extend(timestamps)

qt_times_s = [s / 1000 for s in itertools.accumulate(short_files_rr)]
f_interp_rr = interp1d(short_files_timestamps, short_files_rr, kind='linear')
x_rr = np.arange(min(short_files_timestamps), max(short_files_timestamps)-1, 1/frac_sec)
interpolated_rr = f_interp_rr(x_rr)

f_interp_qt = interp1d(qt_times_s, short_files_qt, kind='linear')
x_qt = np.arange(min(qt_times_s), max(qt_times_s)-1, 1/frac_sec)
interpolated_qt = f_interp_qt(x_qt)
rr = pl.concat([rr, process_rr(interpolated_rr, snip_len=3500)])
qt = pl.concat([qt, process_qt(interpolated_qt, snip_len=3500)])

rr_11 = process_rr(interpolated_rr, snip_len=6000)
qt_11 = process_qt(interpolated_qt, snip_len=6000)

rr.write_csv('processed_data/rr_processed.csv')
qt.write_csv('processed_data/qt_processed.csv')

rr_18.write_csv('processed_data/rr_18_processed.csv')
qt_18.write_csv('processed_data/qt_18_processed.csv')

rr_07.write_csv('processed_data/rr_07_processed.csv')
qt_07.write_csv('processed_data/qt_07_processed.csv')

rr_11.write_csv('processed_data/rr_11_processed.csv')
qt_11.write_csv('processed_data/qt_11_processed.csv')
