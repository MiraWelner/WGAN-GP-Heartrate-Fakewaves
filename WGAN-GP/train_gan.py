"""
Mira Welner
June 2025

This module loads Dr. Cong's ECG data from the proccessed_data.csv, removes outliers, normalizes, etc.
Once dataloaders are created, it uses the second 2/3 of the dataset to train the WGAN-GP. It then stores the
pytorch model, and displays and stores the comparison of the test set and the trained set.
"""

#libraries imported
import pandas as pd
import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan
from scipy.signal import find_peaks

#Constants set here
seconds = 7
hz=500
batch_size=248
latent_dim = 100
epochs = 500
itt = 100
#if you want you can only train on specific signals
signals_tested = ['ecg0_channel0']

#first it makes sure that the machine you are running on has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

#the proccessed data which is proccessed in the 0_Raw_Data_Processing folder is loaded and channels selected
df = pd.read_csv('../Raw_Data_Processing/proccessed_data.csv', index_col=0)
df = df[signals_tested]

#train/test split
train_data = df.iloc[len(df)//3:]
test_data = df.iloc[:len(df)//3]

#snips the data in accordance with the R peak. Since they are all from the same
# patient, the ecg0 sensor is used to determine R peak. Any other ecg singal would likely be similar.
def generate_dataloader(data, seconds, hz=hz):
    peaks, _ = find_peaks(data[signals_tested[0]], height=0.1)
    snips = np.array([data[p:p+seconds*hz].T for p in peaks if len(data[p:p+seconds*hz])==seconds*hz])
    torch_data = torch.from_numpy(snips.astype(np.float32))
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=batch_size, shuffle=False, num_workers=6)
    return dataloader
dl_ecg_train = generate_dataloader(train_data, seconds)
dl_ecg_test = generate_dataloader(test_data, seconds)


#train the model!
generator, discriminator = train_wgan(dl_ecg_train,
    dl_ecg_test,
    latent_dim=latent_dim,
    signal_length=seconds*hz,
    epochs=epochs,
    batch_size=batch_size,
    channels=len(df.columns))
torch.save(generator.state_dict(), f"../models/generator_weights_{'_'.join(signals_tested)}.pth")
#generator = Generator(signal_length=seconds*hz, latent_dim=seconds*hz, channels=len(df.columns)).cuda()
#generator.load_state_dict(torch.load("generator_weights.pth", weights_only=True))

#test models
fake_signals = np.concatenate([generator(torch.randn(1, latent_dim).cuda()).cpu().detach().numpy() for _ in range(itt)], axis=0)
real_test_signal = np.array(next(iter(dl_ecg_train))).squeeze()

#plot test results
_, axes = plt.subplots(len(signals_tested),1, figsize=(17,11))
if len(signals_tested) == 1:
    # When there is only one channel, it breaks some things
    # This is somewhat of a hack but it fixes it
    axes = [axes]
    real_test_signal = np.expand_dims(real_test_signal,axis=1)

xticks = np.arange(0, hz*seconds + 1, hz)
xtick_labels = np.arange(0, seconds + 1, 1)

for i in range(len(signals_tested)):
    #mean and 95% confidence for WGAN-GP output
    wgan_signal = np.mean(fake_signals[:, i, :], axis=0)
    wgan_lower = np.percentile(fake_signals[:, i, :], 2.5, axis=0)
    wgan_upper = np.percentile(fake_signals[:, i, :], 97.5, axis=0)
    wgan_error = np.array([wgan_signal - wgan_lower, wgan_upper - wgan_signal])

    #mean and 95% confidence for all test set
    test_signal = np.mean(real_test_signal[:, i, :], axis=0)
    test_lower = np.percentile(real_test_signal[:, i, :], 2.5, axis=0)
    test_upper = np.percentile(real_test_signal[:, i, :], 97.5, axis=0)
    test_error = np.array([test_signal - test_lower, test_upper - test_signal])

    #plot the error and mean for both test and WGAN set
    axes[i].errorbar(range(0, len(test_signal)), wgan_signal, yerr=test_error,alpha=0.1, color='cornflowerblue')
    axes[i].plot(test_signal, label="Test set - mean and 95% confidence", color='cornflowerblue')

    axes[i].errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=wgan_error,alpha=0.1, color='orange')
    axes[i].plot(wgan_signal, label=f"WGAN-GP generated signal {epochs} epochs - mean and 95% confidence of {itt} iterations", color='orange')

    axes[i].legend()
    axes[i].set_xticks(xticks, xtick_labels)
    axes[i].set_ylim(-1,1)
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Voltage (Normalized)")
    axes[i].set_title(f"Comparison Between Generated {str(df.columns[i]).upper()} Signal and Test Set {str(df.columns[i]).upper()}")

plt.tight_layout()
plt.savefig(f"../figures/output_comparison_{len(signals_tested)}_channels_{epochs}_epochs.png")
plt.show()
