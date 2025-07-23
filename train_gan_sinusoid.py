"""
Mira Welner
July 2025
This script loads the sinusoids generated in generate_sinusoid
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
import polars as pl
from scipy.signal import find_peaks
from typing import NoReturn
plt.rcParams.update({'font.size': 8})

iterations = 50
generated_ouput_iterations = 20

def fft_transform(signal:np.ndarray) -> list[np.ndarray]:
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1.0)
    posfreqs = freqs[:len(signal)//2]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:len(signal)//2]
    peaks = posfreqs[find_peaks(posmag, height=10)[0]]
    return [posfreqs, posmag, peaks]

def weighted_mse(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    weights = np.full_like(a, 1.0, dtype=float)
    return np.sum(weights * (a - b)**2) / np.sum(weights)

def make_train_test_dataloaders(dataset):
    """
    Given a dataset, create a pytorch dataloader out of all of the data. the data is all copies of the same sinusoid which are
    all identical, the test set is idenical to the train dataloader. The goal is for the WGAN-GP to cheat.
    """
    def generate_dataloader(d):
        torch_data = torch.from_numpy(d.astype(np.float32))
        dataloader = DataLoader(TensorDataset(torch_data), batch_size=32, shuffle=False, num_workers=6, drop_last=True)
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

def plot_mean_and_diff(synth_data:list, ground_truth_data:list, full_dataset:pl.DataFrame, name:str, offset:int=0, show=False):
    fig, axes = plt.subplots(5,4,figsize=(18,8))
    for itr, (synthetic, ground_truth, full_wave) in enumerate(zip(synth_data, ground_truth_data, full_dataset.transpose())):
        ground_truth = ground_truth.to_numpy()[0,:]
        wave_ax = axes[itr//2,itr%2*2]
        fft_ax = axes[itr//2,itr%2*2+1]

        wave_ax.plot(full_wave, color='steelblue')
        for synth_wave in synthetic:
            wave_ax.plot(range(offset, offset+len(synth_wave)), synth_wave, color='green', linewidth=0.1)
        synth_mean = f'{np.mean(synthetic):.3f} \u00B1 {np.std(np.mean(synthetic, axis=1)):.2f}'
        synth_std = f'{np.mean(np.std(synthetic, axis=1)):.3f} \u00B1 {np.std(np.std(synthetic, axis=1)):.2f}'

        wave_ax.set_ylim(-1.1,1.1)
        wave_ax.set_yticks([-1,0,1])
        wave_ax.set_title(f'mean:{np.mean(ground_truth):.3f}, std:{np.std(ground_truth):.2f}, \n synth mean: {synth_mean}, synth std: {synth_std}')
        wave_ax.set_ylabel(f"Sinusoid {itr}")

        real_freq, real_mag, real_peaks = fft_transform(ground_truth)
        fft_ax.plot(real_freq, real_mag)
        real_peaks_list = ', '.join(f"{num:.2f}" for num in real_peaks)

        all_gen_peaks = []
        for synth_wave in synthetic:
            synth_freq, synth_mag, synth_peaks = fft_transform(synth_wave)
            fft_ax.plot(synth_freq, synth_mag, color='green', linewidth=0.1)
            all_gen_peaks.append(synth_peaks)

        gen_peaks_mean = np.mean(np.array(all_gen_peaks), axis=0)
        gen_peaks_std = np.std(np.array(all_gen_peaks), axis=0)
        gen_peaks_list = ', '.join(f"{mean:.3f} \u00B1 {std:.2f}" for mean,std in zip(gen_peaks_mean,gen_peaks_std))
        fft_ax.set_title(f'Peak Freq: {real_peaks_list}, \n Synth Peak Freq: {gen_peaks_list}')
        fft_ax.set_xlim(0,0.2)

    fig.tight_layout()
    plt.savefig(f'figures/{name}.png')
    if show:
        plt.show()

def make_table(synth_data:list, ground_truth_data:list, full_dataset:pl.DataFrame, name:str, offset:int=0, show=False) -> NoReturn:
    fig, axes = plt.subplots(5,4,figsize=(18,8))
    for itr, (synthetic, ground_truth, full_wave) in enumerate(zip(synth_data, ground_truth_data, full_dataset.transpose())):
        ground_truth = ground_truth.to_numpy()[0,:]
        wave_ax = axes[itr//2,itr%2*2]
        desc_ax = axes[itr//2,itr%2*2+1]

        wave_ax.plot(full_wave, color='steelblue')
        for synth_wave in synthetic:
            wave_ax.plot(range(offset, offset+len(synth_wave)), synth_wave, color='green', linewidth=0.1)

        _, ground_truth_mag, _ = fft_transform(ground_truth)

        all_synth_mag = []
        for synth_wave in synthetic:
            _, synth_mag, _ = fft_transform(synth_wave)
            all_synth_mag.append(synth_mag)


        wave_ax.set_ylim(-1.1,1.1)
        wave_ax.set_yticks([-1,0,1])
        wave_ax.set_ylabel(f"Sinusoid {itr}")

        desc_ax.text(0.01, 0.95, f'Ground Truth Mean:{np.mean(ground_truth):.3f}, STD:{np.std(ground_truth):.2f}', ha='left', va='top')
        desc_ax.text(0.01, 0.80,f'Synthetic Data Mean: {np.mean(np.mean(synthetic, axis=1)):.3f} \u00B1 {np.std(np.mean(synthetic, axis=1)):.2f}', ha='left', va='top')
        desc_ax.text(0.01, 0.65, f'Synthetic Data STD: {np.mean(np.std(synthetic, axis=1)):.3f} \u00B1 {np.std(np.std(synthetic, axis=1)):.2f}', ha='left', va='top')
        all_mses = [weighted_mse(ground_truth,synth_wave) for synth_wave in synthetic]
        all_fft_mses = [weighted_mse(ground_truth_mag,synth_mag) for synth_mag in all_synth_mag]

        desc_ax.text(0.01, 0.35, f'Weighted Waveform MSE - Mean:{np.mean(all_mses):.5f} \u00B1:{np.std(all_mses):.3f}', ha='left', va='top')
        desc_ax.text(0.01, 0.20, f'Weighted FFT MSE - Mean:{np.mean(all_fft_mses):.5f} \u00B1:{np.std(all_fft_mses):.3f}', ha='left', va='top')


        desc_ax.set_xticks([])
        desc_ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(f"figures/{name}.png")
    if show:
        plt.show()






sine_heartrate = pl.read_csv('processed_data/sinusoid.csv', has_header=False)
first_half = sine_heartrate[:, :300].transpose()
second_half = sine_heartrate[:, 300:].transpose()

dataloaders_firsthalf, testsets_firsthalf = make_train_test_dataloaders(first_half)
_, testsets_secondhalf = make_train_test_dataloaders(second_half)

all_synthetic_outputs = []
for itr, (dataloader, testset) in enumerate(zip(dataloaders_firsthalf, testsets_firsthalf)):
    wgan_gp = train_store_gan(dataloader, signal_length=300, itr=itr, just_load=True)
    synthetic_output = np.array([wgan_gp(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(generated_ouput_iterations)])
    all_synthetic_outputs.append(synthetic_output)

plot_mean_and_diff(all_synthetic_outputs, testsets_firsthalf, sine_heartrate, 'sinusoid_output_first_half', show=True)
make_table(all_synthetic_outputs, testsets_firsthalf, sine_heartrate, 'table_first_half', show=True)

plot_mean_and_diff(all_synthetic_outputs, testsets_secondhalf, sine_heartrate, 'sinusoid_output_second_half', offset = 300, show=True)
make_table(all_synthetic_outputs, testsets_secondhalf, sine_heartrate, 'table_second_half', offset = 300, show=True)
