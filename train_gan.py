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

datatype = 'qt'
dataset = pd.read_csv(f'processed_data/{datatype}_processed.csv', index_col=0)
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
torch.save(generator.state_dict(), f"models/generator_{epochs}_{datatype}.pth")
"""
generator = Generator(signal_length=signal_length, latent_dim=latent_dim, channels=train_data.shape[1]).cuda()
generator.load_state_dict(torch.load(f"models/generator_{epochs}_{datatype}.pth", weights_only=True))

#test models
fake_signal = np.array([generator(torch.randn(1, latent_dim).cuda()).cpu().detach().numpy().squeeze() for _ in range(itt)])
real_test_signal = test_data.squeeze()

#plot the error and mean for both test and WGAN set
wgan_signal = np.mean(fake_signal, axis=0).squeeze()
wgan_std_dev = np.std(fake_signal, axis=0, ddof=1)  #
wgan_median_dev = np.median(fake_signal, axis=0)
#Sample standard deviation
wgan_n = fake_signal.shape[0]  # Sample size
z_score = 1.96  # Z-score for 95% confidence interval
wgan_margin_of_error = z_score * (wgan_std_dev / np.sqrt(wgan_n))
wgan_error = np.array([wgan_margin_of_error, wgan_margin_of_error]).squeeze()

#mean and 95% confidence for all test set
test_signal = np.mean(real_test_signal, axis=0).squeeze()
test_std_dev = np.std(real_test_signal, axis=0, ddof=1)  # Sample standard deviation
test_n = real_test_signal.shape[0]  # Sample size
test_margin_of_error = z_score * (test_std_dev / np.sqrt(test_n))
test_error = np.array([test_margin_of_error, test_margin_of_error]).squeeze()
_ = plt.figure(figsize=(14,6))


#plot the error and mean for both test and WGAN set
plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=test_error,alpha=0.1, color='cornflowerblue')
plt.plot(test_signal, label="Test set — mean and 95% confidence", color='cornflowerblue')

plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=wgan_error,alpha=0.1, color='rebeccapurple')
plt.plot(wgan_signal, label=f"WGAN-GP Output — mean and 95% confidence of {itt} iterations", color='rebeccapurple')

plt.legend()
plt.xlabel("Time (s)")
plt.xticks(np.arange(0, len(wgan_signal)+1, 500), [str(i) for i in np.arange(0, len(wgan_signal)//2+1, 250)])
plt.ylabel("Scaled QT Lengths -- (QT/350)-1")

wgan_mean = np.round(np.mean(wgan_signal),3)
wgan_median = np.round(np.median(wgan_signal),3)
test_mean = np.round(np.mean(test_signal),3)
test_median = np.round(np.median(test_signal),3)

plt.title(f"Comparison Between original QT lengths and WGAN-Generated QT Lengths \n Generated Mean: {wgan_mean:.3f}, Generated Median {wgan_median:.3f}, Original Mean: {test_mean:.3f}, Original Median {test_median:.3f}")

plt.tight_layout()
plt.savefig(f"figures/wgan_comparison/{datatype}_{epochs}.png")
plt.show()
