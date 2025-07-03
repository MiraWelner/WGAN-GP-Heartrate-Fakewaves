"""
Mira Welner
June 2025

This script loads Dr. Dey's heartrate data and proccesses it to be 1/10th of a second, with the y value being the ms in the r-r
distance at that individual decasecond.
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.interpolate import interp1d
from random import randrange

frac_sec = 2 #the x values are half of a second
rr_df = pd.DataFrame()
qt_df = pd.DataFrame()

def process_rr(rr_distance_ms, snip_len = 3500):
    bpm = 60000/rr_distance_ms
    scaled_heartrate = bpm/75 -1
    num_samples = len(scaled_heartrate)//snip_len
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df

def process_qt(qt_distance_ms, snip_len = 3500):
    num_samples = len(qt_distance_ms)//snip_len
    scaled_heartrate_trimmed = qt_distance_ms[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df


def proccess_datafile(path):
    """
    This only works on the 18-58-25 files, and the
    07-15-37 files, or files of similar format
    """
    file = pd.read_csv(path)
    rr_distance_ms = file.iloc[:, 3]
    qt_distance_ms = np.array([0.0 if val == '  ' else val for val in file.iloc[:, 4]], dtype=float)
    x = file.iloc[:,1]
    x_ms = np.arange(min(x), max(x)-1, 1/frac_sec)

    f_interp_rr = interp1d(x, rr_distance_ms, kind='linear')
    f_interp_qt = interp1d(x, qt_distance_ms, kind='linear')
    return process_rr(f_interp_rr(x_ms)), process_qt(f_interp_qt(x_ms))


for i in range(1,5):
    rr_data, qt_data = proccess_datafile(f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv")
    rr_df = pd.concat([rr_df, rr_data])
    qt_df = pd.concat([qt_df, qt_data])


for i in range(1,5):
    rr_data, qt_data = proccess_datafile(f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv")
    rr_df = pd.concat([rr_df, rr_data])
    qt_df = pd.concat([qt_df, qt_data])


# the everyRRQTinputIntoEntropy files are proccessed differently because they are formatted
# differently for some reason
everyRRQT_time_data = []
everyRRQT_rr_data = []
everyRRQT_qt_data = []
for i in range(1,28):
    path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    file = pd.read_csv(path)
    x = file.iloc[:,0]
    if len(everyRRQT_time_data):
        x+=everyRRQT_time_data[-1]

    everyRRQT_rr_data.extend(np.asarray(file.iloc[:,1])*1000)
    everyRRQT_qt_data.extend(np.asarray(file.iloc[:,2])*1000)
    everyRRQT_time_data.extend(x)

f_interp_rr = interp1d(everyRRQT_time_data, everyRRQT_rr_data, kind='linear')
f_interp_qt = interp1d(everyRRQT_time_data, everyRRQT_qt_data, kind='linear')
x_ms = np.arange(min(everyRRQT_time_data), max(everyRRQT_time_data)-1, 1/frac_sec)
rr_df = pd.concat([rr_df, process_rr(f_interp_rr(x_ms))])
qt_df = pd.concat([qt_df, process_rr(f_interp_qt(x_ms))])

rr_df.to_csv('processed_data/rr_processed.csv')
qt_df.to_csv('processed_data/qt_processed.csv')
