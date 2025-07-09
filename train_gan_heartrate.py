"""
Mira Welner
July 2025
This script generates a set of square and triangle waves, then trains the WGAN-GP on them. It varies the
wavelength, waveheight, whether it is evenly spaces, randomly spaced, or pattern spaced, and the wave peak length.

It then saves the figures and models.
"""
import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
import pandas as pd



def make_train_test_dataloaders(rr, qt):
    def generate_dataloader(data):
        torch_data = torch.from_numpy(data.astype(np.float32))
        dataloader = DataLoader(TensorDataset(torch_data), batch_size=256*2, shuffle=False, num_workers=6, drop_last=True)
        return dataloader

    #train/test split
    print(rr.shape)
    train_rr = np.expand_dims(rr.values[rr.shape[0]//8:],1)
    test_rr = np.expand_dims(rr.values[:rr.shape[0]//8],1)
    train_qt = np.expand_dims(qt.values[qt.shape[0]//8:],1)
    test_qt = np.expand_dims(qt.values[:qt.shape[0]//8],1)

    train_rr_dl = generate_dataloader(train_rr)
    train_qt_dl = generate_dataloader(train_qt)

    return train_rr_dl, test_rr, train_qt_dl, test_qt

def train_store_gan(train_rr, train_qt, epochs=500, just_load=False):
    if not just_load:
        gen_rr, _ = train_wgan(train_rr, epochs=epochs)
        torch.save(gen_rr.state_dict(), "models/generator_rr.pth")
        gen_qt, _ = train_wgan(train_qt, epochs=epochs)
        torch.save(gen_qt.state_dict(), "models/generator_qt.pth")

    gen_rr, gen_qt = Generator().cuda(), Generator().cuda()
    gen_rr.load_state_dict(torch.load("models/generator_rr.pth", weights_only=True))
    gen_qt.load_state_dict(torch.load("models/generator_qt.pth", weights_only=True))
    return gen_rr, gen_qt

def plot_mean_and_diff(rr_test_squeeze, rr_gen_output, qt_test_squeeze, qt_gen_output, show=False):
    _, axes = plt.subplots(2,1, figsize=(17,7))

    mean_rr_gen = np.mean(rr_gen_output, axis=0)
    mean_rr_real = np.mean(rr_test_squeeze, axis=0)
    mean_diff = np.mean(mean_rr_real - mean_rr_gen, axis=0)
    lower_bound = mean_rr_real - np.minimum(mean_diff, np.zeros(len(mean_rr_real)))
    upper_bound = mean_rr_real - np.maximum(mean_diff, np.zeros(len(mean_rr_real)))

    #axes[0].fill_between(range(rr_gen_output.shape[1]), lower_bound, upper_bound, alpha=0.5, label="Mean Difference Between Real and Gen")
    axes[0].plot(mean_rr_real, label="Mean Real Data", color='red')
    axes[0].plot(mean_rr_gen, label="Mean Generated Data", color='orange')

    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    rr_gen_mean = np.mean(rr_gen_output)
    rr_gen_median = np.median(rr_gen_output)
    rr_gen_std = np.std(rr_gen_output)
    rr_real_mean = np.mean(rr_test_squeeze)
    rr_real_median = np.median(rr_test_squeeze)
    rr_real_std = np.std(rr_test_squeeze)
    axes[0].set_title(f"Comparison Between original RR intervals and WGAN-Generated RR intervals \n Generated Mean: {rr_gen_mean:.3f}, Generated Median {rr_gen_median:.3f}, Generated STD {rr_gen_std:.3f}, Original Mean: {rr_real_mean:.3f}, Original Median {rr_real_median:.3f}, Original STD {rr_real_std:.3f}")

    #plot triangle wave data on lower plot
    mean_qt_real = np.mean(qt_test_squeeze, axis=0)
    mean_qt_gen = np.mean(qt_gen_output, axis=0)
    mean_diff = np.mean(mean_qt_real - mean_qt_gen, axis=0)
    lower_bound = mean_qt_real + np.minimum(mean_diff, np.zeros(len(mean_qt_real)))
    upper_bound = mean_qt_real + np.maximum(mean_diff, np.zeros(len(mean_qt_real)))

    #axes[1].fill_between(range(qt_gen_output.shape[1]), lower_bound, upper_bound, alpha=0.5, label="Mean Difference Between Real and Gen")
    axes[1].plot(mean_qt_real, label="Mean Real Data", color='red')
    axes[1].plot(mean_qt_gen, label="Mean Generated Data", color='orange')
    axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    qt_gen_mean = np.mean(qt_gen_output)
    qt_gen_median = np.median(qt_gen_output)
    qt_gen_std = np.std(qt_gen_output)
    qt_real_mean = np.mean(qt_test_squeeze)
    qt_real_median = np.median(qt_test_squeeze)
    qt_real_std = np.std(qt_test_squeeze)
    axes[1].set_title(f"Comparison Between original QT intervals and WGAN-Generated QT intervals \n Generated Mean: {qt_gen_mean:.3f}, Generated Median {qt_gen_median:.3f}, Generated STD {qt_gen_std:.3f}, Original Mean: {qt_real_mean:.3f}, Original Median {qt_real_median:.3f}, Original STD {qt_real_std:.3f}")

    plt.tight_layout()
    plt.savefig("figures/wgan_comparison/qt_rr.png")
    if show:
        plt.show()

def plot_examples(rr_gen, qt_gen, show=False):
    _, axes = plt.subplots(2,1, figsize=(17,7))
    axes[0].plot(rr_gen[0], color='cornflowerblue')
    axes[0].set_title("Example of Generated RR Output")
    axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[0].set_xlabel("Time (s)")


    axes[1].plot(qt_gen[0], color='cornflowerblue')
    axes[1].set_title("Example of Generated QT Wave")
    axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("figures/samples/qt_rr.png")
    if show:
        plt.show()


#ensure that the computer has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

qt = pd.read_csv('processed_data/qt_processed.csv', index_col=0)
rr = pd.read_csv('processed_data/rr_processed.csv', index_col=0)


train_rr, test_rr, train_qt, test_qt = make_train_test_dataloaders(rr, qt)
gen_rr, gen_qt = train_store_gan(train_rr, train_qt)

rr_gen_output = np.array([gen_rr(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(500)])
rr_test_squeeze = test_rr.squeeze()

qt_gen_output = np.array([gen_qt(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(500)])
qt_test_squeeze = test_qt.squeeze()

plot_mean_and_diff(rr_test_squeeze, rr_gen_output, qt_test_squeeze, qt_gen_output, show=True)
plot_examples(rr_gen_output, qt_gen_output, show=True)
