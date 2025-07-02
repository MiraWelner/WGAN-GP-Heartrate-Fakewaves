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
from wgan import train_wgan, Generator
import pandas as pd
from random import randrange

#Constants set here
batch_size=64
signal_length = 3500
latent_dim = 100
epochs = 1000
itt = 100


#first it makes sure that the machine you are running on has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

dataset = pd.read_csv('../processed_data/heartrate_processed.csv', index_col=0)
#train/test split
train_data = np.expand_dims(dataset[dataset.shape[0]//3:],1)
test_data = np.expand_dims(dataset[:dataset.shape[0]//3],1)
# snips the data in accordance with the R peak. Since they are all from the same
# patient, the ecg0 sensor is used to determine R peak. Any other ecg singal would likely be similar.
def generate_dataloader(data):
    torch_data = torch.from_numpy(data.astype(np.float32))
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
    return dataloader
dl_ecg_train = generate_dataloader(train_data)
dl_ecg_test = generate_dataloader(test_data)

#train the model!
"""
generator, discriminator = train_wgan(dl_ecg_train,
    dl_ecg_test,
    latent_dim=latent_dim,
    signal_length=signal_length,
    epochs=epochs,
    batch_size=batch_size,
    channels=1)
torch.save(generator.state_dict(), f"../models/generator_weights_heartrate_{epochs}.pth")
"""
generator = Generator(signal_length=signal_length, latent_dim=latent_dim, channels=train_data.shape[1]).cuda()
generator.load_state_dict(torch.load(f"../models/generator_weights_heartrate_{epochs}.pth", weights_only=True))

#test models
fake_signal = np.array([generator(torch.randn(1, latent_dim).cuda()).cpu().detach().numpy().squeeze() for _ in range(itt)])
real_test_signal = test_data.squeeze()

#plot the error and mean for both test and WGAN set
wgan_signal = np.mean(fake_signal, axis=0).squeeze()
wgan_lower = np.percentile(fake_signal, 2.5, axis=0)
wgan_upper = np.percentile(fake_signal, 97.5, axis=0)
wgan_error = np.array([wgan_signal - wgan_lower, wgan_upper - wgan_signal]).squeeze()

#mean and 95% confidence for all test set
test_signal = np.mean(real_test_signal, axis=0).squeeze()
test_lower = np.percentile(real_test_signal, 2.5, axis=0)
test_upper = np.percentile(real_test_signal, 97.5, axis=0)
test_error = np.array([test_signal - test_lower, test_upper - test_signal]).squeeze()

_ = plt.figure(figsize=(14,6))


#plot the error and mean for both test and WGAN set
plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=test_error,alpha=0.1, color='cornflowerblue')
plt.plot(test_signal, label="Test set — mean and 95% confidence", color='cornflowerblue')

plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=wgan_error,alpha=0.1, color='rebeccapurple')
plt.plot(wgan_signal, label=f"WGAN-GP Output — mean and 95% confidence of {itt} iterations", color='rebeccapurple')

plt.legend()
plt.ylim(-0.75,0.75)
plt.xlabel("Time (s)")
plt.xticks(np.arange(0, len(wgan_signal)+1, 500), np.arange(0, len(wgan_signal)//10+1, 50))
plt.ylabel("Normalized BPM")
plt.title(f"Comparison Between Generated Signal and Test Set {epochs} epochs")

plt.tight_layout()
plt.savefig(f"../figures/wgan_comparison/heartrate_{epochs}.png")
plt.show()


for _ in range(10):
    _ = plt.figure(figsize=(14,6))
    randval = randrange(real_test_signal.shape[0])
    plt.plot(real_test_signal[randval,:])
    plt.title("Randomly Selected Heartrate from Test Set")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized BPM")
    plt.xticks(np.arange(0, len(wgan_signal)+1, 500), np.arange(0, len(wgan_signal)//10+1, 50))
    plt.savefig(f"../figures/sample_{randval}.png")

    plt.show()
