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
from random import randrange

#Constants set here
seconds = 7
hz=500
batch_size=4
latent_dim = 100
epochs = 1000
itt = 100


#first it makes sure that the machine you are running on has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

dataset = np.loadtxt('heartrate_data/07-15-37_8_hours_part1_v2_wholecaseRRiQTi.csv', delimiter=',')
print(dataset)
sys.exit()
#train/test split
train_data = np.expand_dims(dataset[10:],1)
test_data = np.expand_dims(dataset[:10],1)
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
    signal_length=seconds*hz,
    epochs=epochs,
    batch_size=batch_size,
    channels=1)
torch.save(generator.state_dict(), f"../models/generator_weights_heartrate_{epochs}.pth")
"""
generator = Generator(signal_length=seconds*hz, latent_dim=latent_dim, channels=train_data.shape[1]).cuda()
generator.load_state_dict(torch.load(f"../models/generator_weights_heartrate_{epochs}.pth", weights_only=True))

#test models
fake_signal = np.stack([generator(torch.randn(1, latent_dim).cuda()).cpu().detach().numpy().squeeze() for _ in range(itt)])
mean_sig = np.mean(fake_signal ,axis=0)
std_sig = np.std(fake_signal, axis=0)
median_sig = np.median(fake_signal, axis=0)

#real_test_signal = test_data[randrange(test_data.shape[1]),:].T
#plot test results
_, axes = plt.subplots(3, 1, figsize=(17,8))

axes[0].plot(mean_sig)
axes[0].set_title("mean WGAN-GP output")

axes[1].plot(std_sig)
axes[1].set_title("std WGAN-GP output")

axes[2].plot(median_sig)
axes[2].set_title("median WGAN-GP output")

xticks = np.arange(0, hz*seconds + 1, hz)
xtick_labels = [str(i) for i in np.arange(0, seconds + 1, 1)]


#plot the error and mean for both test and WGAN set
#plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=test_error,alpha=0.1, color='cornflowerblue')
#plt.plot(fake_signal, label="WGAN-GP generated heart rate", color='cornflowerblue')

#plt.errorbar(range(0, len(wgan_signal)), wgan_signal, yerr=wgan_error,alpha=0.1, color='orange')
#plt.plot(real_test_signal, label="Test Generated Heart Rate", color='orange')
"""
plt.legend()
plt.ylim(-0.25,2)
plt.xlabel("Time")
plt.ylabel("Normalized BPM")
plt.title("Comparison Between Generated Signal and Test Set")
"""
plt.tight_layout()
plt.savefig(f"../figures/wgan_comparison/heartrate_{epochs}.png")
plt.show()
