"""
Mira Welner
June 2025

This module loads Dr. Cong's ECG data from the proccessed_data.csv, removes outliers, normalizes, etc.
Once dataloaders are created, it uses the second 2/3 of the dataset to train the WGAN-GP. It then stores the
pytorch model, and displays and stores the comparison of the test set and the trained set.
"""

#libraries imported
import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan

#Constants set here
seconds = 7
hz=500
batch_size=256
latent_dim = 100
epochs = 5
itt = 100


#first it makes sure that the machine you are running on has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

data_file= 'alive/08_ch2_merged.bin.csv'
#the proccessed data which is proccessed in the 0_Raw_Data_Processing folder is loaded and channels selected
dataset = np.loadtxt(f'../processed_data/{data_file}', delimiter=',')
#train/test split
train_data = np.expand_dims(dataset[len(dataset)//3:, :],1)
test_data = np.expand_dims(dataset[:len(dataset)//3, :],1)

#snips the data in accordance with the R peak. Since they are all from the same
# patient, the ecg0 sensor is used to determine R peak. Any other ecg singal would likely be similar.
def generate_dataloader(data):
    torch_data = torch.from_numpy(data.astype(np.float32))
    print(f"Torch data shape: {torch_data.shape}")
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
    return dataloader
dl_ecg_train = generate_dataloader(train_data)
dl_ecg_test = generate_dataloader(test_data)

#train the model!
generator, discriminator = train_wgan(dl_ecg_train,
    dl_ecg_test,
    latent_dim=latent_dim,
    signal_length=seconds*hz,
    epochs=epochs,
    batch_size=batch_size,
    channels=1)
torch.save(generator.state_dict(), f"../models/generator_weights_{epochs}.pth")
#generator = Generator(signal_length=seconds*hz, latent_dim=seconds*hz, channels=len(df.columns)).cuda()
#generator.load_state_dict(torch.load("generator_weights.pth", weights_only=True))

#test models
fake_signals = np.concatenate([generator(torch.randn(1, latent_dim).cuda()).cpu().detach().numpy() for _ in range(itt)], axis=0)
real_test_signal = np.array(next(iter(dl_ecg_train))).squeeze()

#plot test results
_ = plt.figure(figsize=(17,11))

xticks = np.arange(0, hz*seconds + 1, hz)
xtick_labels = [str(i) for i in np.arange(0, seconds + 1, 1)]


wgan_signal = np.mean(fake_signals, axis=0)
wgan_lower = np.percentile(fake_signals, 2.5, axis=0)
wgan_upper = np.percentile(fake_signals, 97.5, axis=0)
wgan_error = np.array([wgan_signal - wgan_lower, wgan_upper - wgan_signal])

#mean and 95% confidence for all test set
test_signal = np.mean(real_test_signal, axis=0)
test_lower = np.percentile(real_test_signal, 2.5, axis=0)
test_upper = np.percentile(real_test_signal, 97.5, axis=0)
test_error = np.array([test_signal - test_lower, test_upper - test_signal])

#plot the error and mean for both test and WGAN set
print(len(wgan_signal))
print(len(test_error))
plt.errorbar(range(0, len(wgan_signal[0])), wgan_signal, yerr=test_error,alpha=0.1, color='cornflowerblue')
plt.plot(test_signal, label="Test set - mean and 95% confidence", color='cornflowerblue')

plt.errorbar(range(0, len(wgan_signal[0])), wgan_signal, yerr=wgan_error,alpha=0.1, color='orange')
plt.plot(wgan_signal, label=f"WGAN-GP generated signal {epochs} epochs - mean and 95% confidence of {itt} iterations", color='orange')

plt.legend()
plt.xticks(xticks, xtick_labels)
plt.ylim(-1,1)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (Normalized)")
plt.title("Comparison Between Generated  Signal and Test Set")

plt.tight_layout()
plt.savefig(f"../figures/wgan_comparison_{data_file}_{epochs}.png")
plt.show()
