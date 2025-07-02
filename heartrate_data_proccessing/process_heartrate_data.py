"""
Mira Welner
June 2025

This script loads Dr. Cong's PPG and ECG data, removes outliers, and interpolates the PPG data
which is 125 hz to be 500hz like the ECG data.

As of now, this algorithm does NOT remove noise because my previous notch filter method
cannot easily be automated. So, my hope is that the model will remove the noise.
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from random import randrange

frac_sec = 10 #the x values are 100th of a second

def process_rr(rr_distance_ms, snip_len = 3500):
    bpm = 60000/rr_distance_ms
    scaled_heartrate = bpm/75 -1
    num_samples = len(scaled_heartrate)//snip_len
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df


df = pd.DataFrame()
for i in range(1,5):
    path = f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    file = pd.read_csv(path)
    rr_distance_ms = file.iloc[:, 3]
    x = file.iloc[:,1]
    f_interp = interp1d(x, rr_distance_ms, kind='linear')
    x_ms = np.arange(min(x), max(x)-1, 1/frac_sec)
    y_ms = f_interp(x_ms)
    df = pd.concat([df, process_rr(y_ms)])


for i in range(1,5):
    path = f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    file = pd.read_csv(path)
    rr_distance_ms = file.iloc[:, 3]
    x = file.iloc[:,1]
    f_interp = interp1d(x, rr_distance_ms, kind='linear')
    x_ms = np.arange(min(x), max(x)-1, 1/frac_sec)
    y_ms = f_interp(x_ms)
    df = pd.concat([df, process_rr(y_ms)])

heartrate_11_x = []
heartrate_11_y = []
for i in range(1,28):
    path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    file = pd.read_csv(path)
    x = file.iloc[:,0]
    if len(heartrate_11_x):
        x+=heartrate_11_x[-1]
    rr_distance_s = np.asarray(file.iloc[:,1])
    heartrate_11_y.extend(rr_distance_s*1000)
    heartrate_11_x.extend(x)
f_interp = interp1d(heartrate_11_x, heartrate_11_y, kind='linear')
x_ms = np.arange(min(heartrate_11_x), max(heartrate_11_x)-1, 1/frac_sec)
y_ms = f_interp(x_ms)
df = pd.concat([df, process_rr(y_ms)])

for i in range(20):
    plt.plot(df.iloc[randrange(df.shape[0]),:])
    plt.title(df.shape[0])
    plt.show()
df.to_csv('../processed_data/heartrate_processed.csv')
