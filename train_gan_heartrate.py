"""
Mira Welner
July 2025
This script loads the heartrate data generated in generate_processed_heartrate.py and splits it into distributions based on histograms.
It then trains a gan on each distribution and displays it.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 8})

#data processing params
patient_names = '06-31-24', '09-40-14', '10-48-45', '11-03-38', '13-22-23', '14-17-50'
split_locs = [-0.75, -0.555, -0.16], -0.5, -0.1, None ,-0.5,-0.37
snip_len = 2500

#training params
batch_size = 16
test_train_split = 0.8

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

def make_GAN_test_set(dataset:np.ndarray):
    """
    Given a 2d numpy array, create a test set and a training dataloader. They will be the same data because we
    are CHEATING
    """
    dataset=np.expand_dims(dataset,1)
    torch_data = torch.from_numpy(dataset.astype(np.float32).T)
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
    test_set = dataset.astype(np.float32).T
    return dataloader, test_set

def train_store_gan(train, name, just_load, epochs):
    """
    if just_load, then we just load the generator and return it.
    If just_load is false, then train the new GAN and store it with
    the same naming conventions.
    """
    gan_name = f"models/generator_heartrate_{name}.pth"

    if not just_load:
        gen, _ = train_wgan(train, epochs=epochs, signal_length=snip_len)
        torch.save(gen.state_dict(), gan_name)

    gen = Generator(signal_length=snip_len, latent_dim=100).cuda()
    gen.load_state_dict(torch.load(gan_name, weights_only=True))
    return gen

def split_patient(patient_name:str, split_loc:float, stride:int=snip_len//2):
    """
    split the data into two sets, one of which is a set of continuious snips greater than split_loc
    and the other is lower
    """
    heartrate_data_raw = np.loadtxt(f'processed_data/heartrate_{patient_name}.csv',delimiter=',')
    heartrate_data = heartrate_data_raw[:-2] #remove min and max
    heartrate_min = heartrate_data_raw[-2]
    heartrate_max = heartrate_data_raw[-1]
    segments = np.lib.stride_tricks.sliding_window_view(heartrate_data, window_shape=snip_len)
    segments = segments[::stride]
    # Boolean masks
    mask_gt = np.all(segments > split_loc, axis=1)
    mask_lt = np.all(segments < split_loc, axis=1)

    # Split into two arrays
    segments_gt = segments[mask_gt]
    segments_lt = segments[mask_lt]

    return segments_gt.T, segments_lt.T, heartrate_min, heartrate_max

def get_synthetic_outputs(gan, iterations, latent_dim=100):
    """
    Given a Pytorch GAN model of latent_dim 100, run the gan iterations times and put the result
    in a numpy array which is returned
    """
    synthetic_output = np.array([gan(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(iterations)])
    return synthetic_output

def final_plot(synth_data:list, ground_truth_data:list, data_max:float, data_min:float, name:str, alpha=0.3):
    fig, axes = plt.subplots(len(synth_data),3,figsize=(18,8),constrained_layout=True)
    for itr, (synthetic, ground_truth) in enumerate(zip(synth_data, ground_truth_data)):
        synthetic =   ((synthetic + 1) / 2) * (data_max - data_min) + data_min
        ground_truth =  ((ground_truth + 1) / 2) * (data_max - data_min) + data_min

        wave_ax = axes[itr,0]
        fft_ax = axes[itr,1]
        text_ax = axes[itr,2]

        all_gt_fft = []
        all_synth_fft = []

        for it, gt_wave in enumerate(ground_truth):
            gt_squeezed = gt_wave.squeeze()
            if it==0:
                wave_ax.plot(np.cumsum(gt_squeezed), gt_squeezed, color='blue', alpha=alpha, label=f'Ground Truth: {len(ground_truth)} samples')
            else:
                wave_ax.plot(np.cumsum(gt_squeezed), gt_squeezed, alpha=alpha, linewidth=0.5, color='blue')
            freq, mag, _ = fft_transform(gt_squeezed)
            fft_ax.plot(freq[1:], mag[1:], linewidth=0.5, alpha=alpha, color='blue')
            all_gt_fft.append(mag)

        for it, synth_wave in enumerate(synthetic):
            if it == 0:
                wave_ax.plot(np.cumsum(synth_wave), synth_wave, color='red', alpha=alpha, label=f'Synthetic: {len(synthetic)} samples')
            else:
                wave_ax.plot(np.cumsum(synth_wave), synth_wave, alpha=alpha, linewidth=alpha, color='red')
            synth_freq, synth_mag, synth_peaks = fft_transform(synth_wave)
            fft_ax.plot(synth_freq[1:], synth_mag[1:], linewidth=alpha, alpha=alpha, color='red')
            all_synth_fft.append(synth_mag)

        synth_mean = f'{np.mean(synthetic):.3f} \u00B1 {np.std(np.mean(synthetic, axis=1)):.2f}'
        synth_std = f'{np.mean(np.std(synthetic, axis=1)):.3f} \u00B1 {np.std(np.std(synthetic, axis=1)):.2f}'

        gt_mean = f'{np.mean(ground_truth):.3f} \u00B1 {np.std(np.mean(ground_truth, axis=2)):.2f}'
        gt_std = f'{np.mean(np.std(ground_truth, axis=2)):.3f} \u00B1 {np.std(np.std(ground_truth, axis=2)):.2f}'

        wave_ax.set_ylabel("R-R Interval (s)")
        wave_ax.set_title("R-R Interval (s)")
        wave_ax.legend()

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

        if itr==0:
            fft_ax.set_title("Heartrate FFT")
            wave_ax.set_title("Scaled R-R Distances")
            wave_ax.set_xlabel("Time (s)")

        else:
            wave_ax.set_xticks([])
            fft_ax.set_xticks([])

    plt.suptitle(name)
    plt.savefig(f'figures/{name}.png')
    plt.show()

high_data, low_data, data_min, data_max = split_patient(patient_name = '14-17-50', split_loc= -0.37)
high_data_dl, high_data_test = make_GAN_test_set(high_data)
low_data_dl, low_data_test = make_GAN_test_set(low_data)
wgan_gp_high = train_store_gan(high_data_dl, 'high_dataset', just_load=False, epochs=5000)
wgan_gp_low = train_store_gan(low_data_dl, 'low_dataset', just_load=False, epochs=5000)
synth_high = get_synthetic_outputs(wgan_gp_high, iterations=50)
synth_low = get_synthetic_outputs(wgan_gp_low, iterations=25)
final_plot([synth_high, synth_low], [high_data_test,low_data_test], data_max=data_max, data_min=data_min, name='Patient 14-17-50 R-R distances')
