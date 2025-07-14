"""
Mira Welner
July 2025
This script loads the processed heartrate, splits it such that the first 5 minutes is used to train
the generator and the descriminator tries to distinguish the generated data from the second 5 minutes.
"""

import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
import polars as pl

def make_train_test_dataloaders(rr):
    def generate_dataloader(data):
        torch_data = torch.from_numpy(data.astype(np.float32))
        dataloader = DataLoader(TensorDataset(torch_data), batch_size=256*2, shuffle=False, num_workers=6, drop_last=True)
        return dataloader

    #train/test split
    train_rr = np.expand_dims(rr[rr.shape[0]//8:],1)
    test_rr = np.expand_dims(rr[:rr.shape[0]//8],1)

    train_rr_dl = generate_dataloader(train_rr)

    return train_rr_dl, test_rr

def train_store_gan(train_rr, signal_length, epochs=500, just_load=False):
    if not just_load:
        gen_rr, _ = train_wgan(train_rr, signal_length=signal_length, epochs=epochs)
        torch.save(gen_rr.state_dict(), "models/generator_rr.pth")

    gen_rr = Generator(signal_length=signal_length).cuda()
    gen_rr.load_state_dict(torch.load("models/generator_rr.pth", weights_only=True))
    return gen_rr

def plot_mean_and_diff(rr_test_squeeze, rr_gen_output, show=False):
    _ = plt.figure(figsize=(17,7))

    mean_rr_gen = np.mean(rr_gen_output, axis=0)
    mean_rr_real = np.mean(rr_test_squeeze, axis=0)


    #axes[0].fill_between(range(rr_gen_output.shape[1]), lower_bound, upper_bound, alpha=0.5, label="Mean Difference Between Real and Gen")
    plt.plot(mean_rr_real, label="Mean Real Data", color='red')
    plt.plot(mean_rr_gen, label="Mean Generated Data", color='orange')
    plt.legend()
    plt.xlabel("Time (m)")
    plt.ylabel("Scaled RR Invervals")
    plt.xticks(np.arange(0, 3501, 600), [str(i) for i in range(6)])
    rr_gen_mean = np.mean(rr_gen_output)
    rr_gen_median = np.median(rr_gen_output)
    rr_gen_std = np.std(rr_gen_output)
    rr_real_mean = np.mean(rr_test_squeeze)
    rr_real_median = np.median(rr_test_squeeze)
    rr_real_std = np.std(rr_test_squeeze)
    plt.title(f"Heartrate Over Time (RR Invervals) \n Generated Mean: {rr_gen_mean:.3f}, Generated Median {rr_gen_median:.3f}, Generated STD {rr_gen_std:.3f}, Original Mean: {rr_real_mean:.3f}, Original Median {rr_real_median:.3f}, Original STD {rr_real_std:.3f}")

    plt.tight_layout()
    plt.savefig("figures/wgan_comparison/future_rr.png")
    if show:
        plt.show()

def plot_examples(rr_gen, show=False):
    _ = plt.figure(figsize=(17,7))
    plt.plot(rr_gen[0], color='cornflowerblue')
    plt.title("Example of Generated Heartrate")
    plt.xticks(np.arange(0, 3501, 600), [str(i) for i in range(6)])
    plt.xlabel("Time (m)")
    plt.ylabel("Scaled RR Interval")

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

rr = pl.read_csv('processed_data/rr_processed.csv')

train_rr, test_rr = make_train_test_dataloaders(rr)
gen_rr = train_store_gan(train_rr, signal_length=3500)

rr_gen_output = np.array([gen_rr(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(500)])
rr_test_squeeze = test_rr.squeeze()

plot_mean_and_diff(rr_test_squeeze, rr_gen_output, show=True)
plot_examples(rr_gen_output, show=True)
