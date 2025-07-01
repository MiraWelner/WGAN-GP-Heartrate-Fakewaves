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

def process_rr(rr_distance_ms):
    bpm = 60000/rr_distance_ms
    scaled_heartrate = (bpm/75 -1).to_numpy()
    num_samples = len(scaled_heartrate)//3500
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*3500]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, 3500)
    snip_df = pd.DataFrame(heartrate_snips)
    return snip_df

df = pd.DataFrame()

for path in glob("heartrate_data/07-15-37_8_hours_part*_v2_wholecaseRRiQTi.csv"):
    file_df = pd.read_csv(path)
    rr_distance_ms = file_df.iloc[:, 3]
    snip_df = process_rr(rr_distance_ms)
    df = pd.concat([df, snip_df], axis=0)
    print(df.shape)


for path in glob("heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel*_Abs*.csv"):
    file_df = pd.read_csv(path)
    rr_distance_s = file_df.iloc[:, 1]#.astype(float)
    rr_distance_ms = rr_distance_s*1000
    snip_df = process_rr(rr_distance_ms)
    df = pd.concat([df, snip_df], axis=0)
    print(df.shape)
