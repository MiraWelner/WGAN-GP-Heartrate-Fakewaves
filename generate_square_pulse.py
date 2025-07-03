from scipy.signal import square
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t = np.linspace(0, 1*7, 500*7, endpoint=False)
square_wave = square(2 * np.pi * 5 * t)
fig = plt.figure(figsize=(15,10))
plt.plot(square_wave[:1000])
plt.show()

df = pd.concat([pd.Series(square_wave) for _ in range(10000)], axis=1).T
print(df.shape)
df.to_csv('processed_data/square_wave.csv')
