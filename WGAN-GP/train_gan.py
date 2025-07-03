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
batch_size=256*4
latent_dim = 100
epochs = 500
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
print(test_data.shape)
#snips the data in accordance with the R peak. Since they are all from the same
# patient, the ecg0 sensor is used to determine R peak. Any other ecg singal would likely be similar.
def generate_dataloader(data):
    torch_data = torch.from_numpy(data.astype(np.float32))
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


wgan_signal = np.mean(fake_signals, axis=0).squeeze()
wgan_std_dev = np.std(fake_signals, axis=0, ddof=1)  # Sample standard deviation
wgan_n = fake_signals.shape[0]  # Sample size
z_score = 1.96  # Z-score for 95% confidence interval
wgan_margin_of_error = z_score * (wgan_std_dev / np.sqrt(wgan_n))
wgan_error = np.array([wgan_margin_of_error, wgan_margin_of_error]).squeeze()

#mean and 95% confidence for all test set
test_signal = np.mean(real_test_signal, axis=0).squeeze()
test_std_dev = np.std(real_test_signal, axis=0, ddof=1)  # Sample standard deviation
test_n = real_test_signal.shape[0]  # Sample size
test_margin_of_error = z_score * (test_std_dev / np.sqrt(test_n))
test_error = np.array([test_margin_of_error, test_margin_of_error]).squeeze()

#plot the error and mean for both test and WGAN set
plt.errorbar(range(0, len(test_signal)), test_signal, yerr=test_error, alpha=0.3, color='cornflowerblue', label="Test set - mean and 95% confidence")

plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=wgan_error, alpha=0.3, color='orange', label=f"WGAN-GP generated signal {epochs} epochs - mean and 95% confidence of {itt} iterations")

plt.legend()
plt.xticks(xticks, xtick_labels)
plt.ylim(-1,1)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (Normalized)")
plt.title("Comparison Between Generated Signal and Test Set")
plt.savefig(f"../figures/wgan_comparison/{data_file.replace('/','_')}_{epochs}.png")
plt.show()
