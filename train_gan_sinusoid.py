"""
Mira Welner
July 2025
This script loads the sinusoids generated in generate_sinusoid
"""

import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
import polars as pl
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 7})


iterations = 300
generated_ouput_iterations = 20

def fft_transform(signal:np.ndarray) -> list[np.ndarray]:
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1.0)
    posfreqs = freqs[:len(signal)//2 + 1]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:len(signal)//2 + 1]
    peaks = posfreqs[find_peaks(posmag, height=10)[0]]
    return [posfreqs, posmag, peaks]

def make_train_test_dataloaders(dataset):
    """
    Given a dataset, create a pytorch dataloader out of all of the data. Since the data is all copies of the same sinusoid which are
    all identical, the test set is idenical to the train dataloader. The goal is for the WGAN-GP to cheat.
    """
    def generate_dataloader(d):
        torch_data = torch.from_numpy(d.astype(np.float32))
        dataloader = DataLoader(TensorDataset(torch_data), batch_size=256, shuffle=False, num_workers=6, drop_last=True)
        return dataloader

    #train/test split
    dataloaders = []
    testsets = []
    for series in dataset.iter_columns():
        df = pl.DataFrame({f"col_{i}": series for i in range(iterations)}).transpose()
        testsets.append(df)
        train = np.expand_dims(df,1)
        train_dl = generate_dataloader(train)
        dataloaders.append(train_dl)

    return dataloaders, testsets

def train_store_gan(train, signal_length, itr, just_load, epochs=1000):
    if not just_load:
        gen, _ = train_wgan(train, signal_length=signal_length, epochs=epochs)
        torch.save(gen.state_dict(), f"models/generator_sine_{itr}.pth")

    gen = Generator(signal_length=signal_length).cuda()
    gen.load_state_dict(torch.load(f"models/generator_sine_{itr}.pth", weights_only=True))
    return gen

def plot_mean_and_diff(gen_outputs:list, tests:pl.DataFrame, show=False):
    fig, axes = plt.subplots(5,4,figsize=(17,9))
    for itr, (gen, real) in enumerate(zip(gen_outputs, tests.transpose())):
        real = real.to_numpy()
        wave_ax = axes[itr//2,itr%2*2]
        fft_ax = axes[itr//2,itr%2*2+1]

        wave_ax.plot(real, color='steelblue')
        for gen_wave in gen:
            wave_ax.plot(gen_wave, color='green', linewidth=0.1)
        synth_mean = f'{np.mean(gen):.2f} \u00B1 {np.std(np.mean(gen, axis=1)):.2f}'
        synth_std = f'{np.mean(np.std(gen, axis=1)):.2f} \u00B1 {np.std(np.std(gen, axis=1)):.2f}'

        wave_ax.set_ylim(-1.1,1.1)
        wave_ax.set_yticks([-1,0,1])
        wave_ax.set_title(f'mean:{np.mean(real):.2f}, std:{np.std(real):.2f}, \n synth mean: {synth_mean}, synth std: {synth_std}')
        wave_ax.set_ylabel(f"Sinusoid {itr}")

        real_freq, real_mag, real_peaks = fft_transform(real)
        fft_ax.plot(real_freq, real_mag)
        real_peaks_list = ', '.join(f"{num:.3f}" for num in real_peaks)

        all_gen_peaks = []
        for gen_wave in gen:
            gen_freq, gen_mag, gen_peaks = fft_transform(gen_wave)
            fft_ax.plot(gen_freq, gen_mag, color='green', linestyle=':', linewidth=0.5)
            all_gen_peaks.append(gen_peaks)
        gen_peaks_mean = np.mean(np.array(all_gen_peaks), axis=0)
        gen_peaks_std = np.std(np.array(all_gen_peaks), axis=0)

        gen_peaks_list = ', '.join(f"{mean:.3f} \u00B1 {std:.3f}" for mean,std in zip(gen_peaks_mean,gen_peaks_std))
        fft_ax.set_title(f'Peak Freq: {real_peaks_list}, \n Synth Peak Freq: {gen_peaks_list}')
        fft_ax.set_xlim(0,0.2)

    fig.tight_layout()
    plt.savefig('figures/sinusoid_output_first_half.png')
    plt.show()


sine_heartrate = pl.read_csv('processed_data/sinusoid.csv', has_header=False)
first_half = sine_heartrate[:, :300].transpose()
dataloaders, testsets = make_train_test_dataloaders(first_half)
gen_outputs = []
for itr, (dataloader, testset) in enumerate(zip(dataloaders, testsets)):
    gen = train_store_gan(dataloader, signal_length=300, itr=itr, just_load=True)
    gen_output = np.array([gen(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(generated_ouput_iterations)])
    gen_outputs.append(gen_output)

plot_mean_and_diff(gen_outputs, sine_heartrate, show=True)
#(gen_outputs, testsets)
