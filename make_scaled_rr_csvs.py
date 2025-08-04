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

def scale_data(x:np.ndarray):
    x_scaled = 2*(x - x.min()) / (x.max() - x.min())-1
    return x_scaled, x.min(), x.max()

def process_patient_data(name:str, parts:int):
    """
    Save the data corresponding to the input name in a csv file. The format of the file will go:
    scaled data ; min value ; max value
    The reason that the min and max are saved is so that the data can be un-scaled after it is
    used to train the GAN which requires the scaling
    """
    all_rr = np.array([])
    for i in range(1,parts+1):
        for j in range(len(glob.glob(f"heartrate_data/{name}_24_hours_part{i}_v3_everyRRQTinputIntoEntropy_*.csv"))):
            f = glob.glob(f"heartrate_data/{name}_24_hours_part{i}_v3_everyRRQTinputIntoEntropy_Rel{j+1}_Abs*.csv")[0]
            data_file = np.genfromtxt(f, skip_header=1, usecols=([1]), delimiter=',')
            clipped_datafile = np.clip(data_file, a_min=0, a_max=1.25)
            all_rr = np.append(all_rr,clipped_datafile)
    x_scaled, min, max = scale_data(all_rr)


    np.savetxt(f"processed_data/heartrate_{name}.csv", np.append(x_scaled,np.array([min,max])), delimiter = ",")

process_patient_data('06-31-24', parts=2)
process_patient_data('09-40-14', parts=5)
process_patient_data('10-48-45', parts=4)
process_patient_data('11-03-38', parts=5)
process_patient_data('13-22-23', parts=4)
process_patient_data('14-17-50', parts=4)
