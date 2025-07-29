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
from scipy.signal import find_peaks
import sys
plt.rcParams.update({'font.size': 8})


iterations = 100
generated_ouput_iterations = 20
batch_size = 64*8

def fft_transform(signal:np.ndarray) -> list[np.ndarray]:
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1.0)
    posfreqs = freqs[:len(signal)//2]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:len(signal)//2].flatten()
    peaks = posfreqs[find_peaks(posmag, height=10)[0]]
    return [posfreqs, posmag, peaks]

def weighted_mse(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    weights = np.full_like(a, 1.0, dtype=float)
    return np.sum(weights * (a - b)**2) / np.sum(weights)

def make_train_test_dataloaders(dataset):
    dataset=np.expand_dims(dataset,1)
    torch_data = torch.from_numpy(dataset.astype(np.float32).T)
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
    test_set = dataset.astype(np.float32).T
    return dataloader, test_set

def train_store_gan(train, name, just_load, signal_length=2500, epochs=1000):
    if not just_load:
        gen, _ = train_wgan(train, epochs=epochs, signal_length=signal_length)
        torch.save(gen.state_dict(), f"models/generator_heartrate_{name}.pth")

    gen = Generator(signal_length=signal_length, latent_dim=100).cuda()
    gen.load_state_dict(torch.load(f"models/generator_heartrate_{name}.pth", weights_only=True))
    return gen

def plot_mean_and_diff(synth_data:list, ground_truth_data:list, patient_names:list[str], name:str, show=False):
    fig, axes = plt.subplots(3,3,figsize=(18,8),constrained_layout=True)
    for itr, (synthetic, ground_truth) in enumerate(zip(synth_data, ground_truth_data)):
        wave_ax = axes[itr,0]
        fft_ax = axes[itr,1]
        text_ax = axes[itr,2]

        all_gt_fft = []
        all_synth_fft = []

        for it, gt_wave in enumerate(ground_truth):
            if it==0:
                wave_ax.plot(gt_wave.squeeze().flatten(), linewidth=0.1, color='steelblue', label='Ground Truth')
            else:
                wave_ax.plot(gt_wave.squeeze().flatten(), linewidth=0.1, color='steelblue')
            freq, mag, _ = fft_transform(gt_wave.squeeze())
            fft_ax.plot(freq, mag, color='steelblue', linewidth=0.1)
            all_gt_fft.append(mag)

        for it, synth_wave in enumerate(synthetic):
            if it == 0:
                wave_ax.plot(range(len(synth_wave)), synth_wave, color='green', linewidth=0.1, label="Synthetic")
            else:
                wave_ax.plot(range(len(synth_wave)), synth_wave, color='green', linewidth=0.1)
            synth_freq, synth_mag, synth_peaks = fft_transform(synth_wave)
            fft_ax.plot(synth_freq, synth_mag, color='green', linewidth=0.1)
            all_synth_fft.append(synth_mag)

        synth_mean = f'{np.mean(synthetic):.3f} \u00B1 {np.std(np.mean(synthetic, axis=1)):.2f}'
        synth_std = f'{np.mean(np.std(synthetic, axis=1)):.3f} \u00B1 {np.std(np.std(synthetic, axis=1)):.2f}'

        gt_mean = f'{np.mean(ground_truth):.3f} \u00B1 {np.std(np.mean(ground_truth, axis=2)):.2f}'
        gt_std = f'{np.mean(np.std(ground_truth, axis=2)):.3f} \u00B1 {np.std(np.std(ground_truth, axis=2)):.2f}'

        #wave_ax.set_ylim(-1.1,1.1)
        wave_ax.set_ylabel(f"Patient {patient_names[itr]}")
        wave_ax.set_yticks([-1,0,1])

        text_ax.text(0.01, 0.95, f'Ground Truth Mean:{gt_mean}', ha='left', va='top')
        text_ax.text(0.01, 0.80, f'Ground Truth STD:{gt_std}', ha='left', va='top')
        text_ax.text(0.01, 0.65,f'Synthetic Data Mean: {synth_mean}', ha='left', va='top')
        text_ax.text(0.01, 0.50, f'Synthetic Data STD: {synth_std}', ha='left', va='top')
        #all_mses = [weighted_mse(ground_truth,synth_wave) for gt, synth_wave in zip(ground_truth,synthetic)]
        #all_fft_mses = [weighted_mse(gt_fft,synth_fft) for gt_fft, synth_fft in zip(all_gt_fft, all_synth_fft)]

        #text_ax.text(0.01, 0.35, f'Weighted Waveform MSE - Mean:{np.mean(all_mses):.5f} \u00B1:{np.std(all_mses):.3f}', ha='left', va='top')
        #text_ax.text(0.01, 0.20, f'Weighted FFT MSE - Mean:{np.mean(all_fft_mses):.5f} \u00B1:{np.std(all_fft_mses):.3f}', ha='left', va='top')


        text_ax.set_xticks([])
        text_ax.set_yticks([])

        fft_ax.set_xlim(0,0.1)

        if itr==0:
            fft_ax.set_title("Heartrate FFT")
            wave_ax.set_title("Scaled Heartrate Waveform")
            wave_ax.set_xlabel("Time (s)")

        else:
            wave_ax.set_xticks([])
            fft_ax.set_xticks([])

    plt.suptitle(name)
    plt.savefig(f'figures/{name}.png')
    if show:
        plt.show()


def get_dataset_and_synth(path:str):
    heartrate = np.loadtxt(path,delimiter=',')
    first_half= heartrate[:, :heartrate.shape[1]//2].transpose()
    second_half = heartrate[:, heartrate.shape[1]//2:].transpose()
    first_half_a = first_half[:int(150000//5000),:]
    first_half_b = first_half[int(150000//5000):int(300000//5000),:]
    first_half_c = first_half[int(300000//5000):,:]

    second_half_a = second_half[:int(150000//5000),:]
    second_half_b = second_half[int(150000//5000):int(300000//5000),:]
    second_half_c = second_half[int(300000//5000):,:]


    dl_firsthalf_a, testset_firsthalf_a = make_train_test_dataloaders(first_half_a)
    _, testset_secondhalf_a = make_train_test_dataloaders(second_half_a)

    dataset_a = (dl_firsthalf_a, testset_firsthalf_a, testset_secondhalf_a)

    dl_firsthalf_b, testset_firsthalf_b = make_train_test_dataloaders(first_half_b)
    _, testset_secondhalf_b = make_train_test_dataloaders(second_half_b)

    dataset_b = (dl_firsthalf_b, testset_firsthalf_b, testset_secondhalf_b)

    dl_firsthalf_c, testset_firsthalf_c = make_train_test_dataloaders(first_half_c)
    _, testset_secondhalf_c = make_train_test_dataloaders(second_half_c)

    dataset_c = (dl_firsthalf_c, testset_firsthalf_c, testset_secondhalf_c)

    return dataset_a, dataset_b, dataset_c

dataset_a, dataset_b, dataset_c = get_dataset_and_synth('processed_data/heartrate_07.csv')


"""
dl_11_firsthalf, testset_11_firsthalf, testset_11_secondhalf = get_dataset_and_synth('processed_data/heartrate_11.csv')
dl_18_firsthalf, testset_18_firsthalf, testset_18_secondhalf = get_dataset_and_synth('processed_data/heartrate_18.csv')

dataloaders = [dl_07_firsthalf, dl_11_firsthalf, dl_18_firsthalf]
first_half_tests = [testset_07_firsthalf, testset_11_firsthalf, testset_18_firsthalf]
second_half_tests = [testset_07_secondhalf, testset_11_secondhalf, testset_18_secondhalf]
"""
synthetic_outputs = []
for dataset, name in zip((dataset_a, dataset_b, dataset_c), ['07', '07', '07']):
    dataloader = dataset[0]
    wgan_gp = train_store_gan(dataloader, name, just_load=False)
    synthetic_output = np.array([wgan_gp(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(generated_ouput_iterations)])
    synthetic_outputs.append(synthetic_output)



plot_mean_and_diff(synthetic_outputs, [d[1] for d in (dataset_a, dataset_b, dataset_c)], ['part_a', 'part_b', 'part_c'], name='Patient 07 at 3 different times',  show=True)
