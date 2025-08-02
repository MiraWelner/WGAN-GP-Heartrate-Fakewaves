import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pyedflib

import mne
file = "13-20-30.EDF"
data = mne.io.read_raw_edf(file)
raw_data = np.array(data.get_data())

_ = plt.figure()
print(raw_data.shape)
plt.plot(raw_data.T[:200,:])
plt.show()
"""
heartrate_data = np.genfromtxt('13-20-30.EDF',delimiter=',',usecols=([1]),skip_header=1)
heartrate_data = np.clip(heartrate_data, a_max = 1.2, a_min=None)
_ = plt.figure()
plt.plot(heartrate_data)
plt.show()
"""
