import numpy as np
import torch
import matplotlib.pyplot as plt
from wgan import train_wgan, Generator
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from torch.utils.data import TensorDataset, DataLoader


patient_names = '06-31-24', '09-40-14', '10-48-45', '11-03-38', '13-22-23', '14-17-50'
snip_len=60*60
batch_size = 16


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

def fft_transform(signal:np.ndarray) -> list[np.ndarray]:
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1.0)
    posfreqs = freqs[:len(signal)//2]
    magnitude = np.abs(fft_result)
    posmag =  magnitude[:len(signal)//2].flatten()
    peaks = posfreqs[find_peaks(posmag, height=10)[0]]
    return [posfreqs, posmag, peaks]

def split_data(patient):
    heartrate_raw = np.loadtxt(f'processed_data/heartrate_{patient}.csv', delimiter=',')
    heartrate_data = heartrate_raw[:-2] #remove min and max
    heartrate_min = heartrate_raw[-2]
    heartrate_max = heartrate_raw[-1]
    total_len = len(heartrate_data) // snip_len * snip_len  # truncate to multiple of snip_len
    clipped = heartrate_data[:total_len]
    shaped_data = clipped.reshape(-1, snip_len).T
    first_half = shaped_data[:int(snip_len/2),:]
    second_half = shaped_data[int(snip_len/2):,:]
    return first_half, second_half, heartrate_min, heartrate_max

def train_store_gan(train, name, just_load, epochs):
    """
    if just_load, then we just load the generator and return it.
    If just_load is false, then train the new GAN and store it with
    the same naming conventions.
    """
    gan_name = f"models/generator_heartrate_{name}.pth"
    if not just_load:
        gen, _ = train_wgan(train, epochs=epochs, signal_length=int(snip_len//2))
        torch.save(gen.state_dict(), gan_name)

    gen = Generator(signal_length=int(snip_len//2), latent_dim=100).cuda()
    gen.load_state_dict(torch.load(gan_name, weights_only=True))
    return gen

def get_synthetic_outputs(gan, iterations, latent_dim=100):
    """
    Given a Pytorch GAN model of latent_dim 100, run the gan iterations times and put the result
    in a numpy array which is returned
    """
    synthetic_output = np.array([gan(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(iterations)])
    return synthetic_output

def final_plot(synth_data:list, ground_truth_data:list, data_max:float, data_min:float, name:str, alpha=0.1):
    fig, axes = plt.subplots(len(synth_data),3,figsize=(18,8),constrained_layout=True)
    for itr, (synthetic, ground_truth) in enumerate(zip(synth_data, ground_truth_data)):
        synthetic =   ((synthetic + 1) / 2) * (data_max - data_min) + data_min
        ground_truth =  ((ground_truth + 1) / 2) * (data_max - data_min) + data_min

        wave_ax = axes[itr,0]
        fft_ax = axes[itr,1]
        fft_ax.set_ylim(0,10)
        wave_ax.set_ylim(0.5,1.2)
        text_ax = axes[itr,2]

        for it, synth_wave in enumerate(synthetic):
            wave_ax.plot(np.cumsum(synth_wave), synth_wave, alpha=alpha, color='red')
            synth_freq, synth_mag, synth_peaks = fft_transform(synth_wave)
            fft_ax.plot(synth_freq[1:], synth_mag[1:], alpha=alpha, color='red')

        for it, gt_wave in enumerate(ground_truth):
            gt_squeezed = gt_wave.squeeze()
            wave_ax.plot(np.cumsum(gt_squeezed), gt_squeezed, alpha=alpha, color='blue')
            freq, mag, _ = fft_transform(gt_squeezed)
            fft_ax.plot(freq[1:], mag[1:], alpha=alpha, color='blue')

        synth_mean = f'{np.mean(synthetic):.3f} \u00B1 {np.std(np.mean(synthetic, axis=1)):.2f}'
        synth_std = f'{np.mean(np.std(synthetic, axis=1)):.3f} \u00B1 {np.std(np.std(synthetic, axis=1)):.2f}'

        gt_mean = f'{np.mean(ground_truth):.3f} \u00B1 {np.std(np.mean(ground_truth, axis=2)):.2f}'
        gt_std = f'{np.mean(np.std(ground_truth, axis=2)):.3f} \u00B1 {np.std(np.std(ground_truth, axis=2)):.2f}'

        wave_ax.set_ylabel("R-R Interval (s)")
        wave_ax.set_title(f"Scaled R-R Distances, part {itr}")

        legend_lines = [Line2D([0], [0], color='blue'),  Line2D([0], [0], color='red')]
        wave_ax.legend(legend_lines, [f"Ground Truth: {len(ground_truth)} samples", f"Synthetic: {len(synthetic)} samples"])

        text_ax.text(0.01, 0.95, f'Ground Truth Mean:{gt_mean}', ha='left', va='top')
        text_ax.text(0.01, 0.80, f'Ground Truth STD:{gt_std}', ha='left', va='top')
        text_ax.text(0.01, 0.65,f'Synthetic Data Mean: {synth_mean}', ha='left', va='top')
        text_ax.text(0.01, 0.50, f'Synthetic Data STD: {synth_std}', ha='left', va='top')

        text_ax.set_xticks([])
        text_ax.set_yticks([])


        if itr==0:
            fft_ax.set_title("Heartrate FFT")
            wave_ax.set_xlabel("Time (s)")

        else:
            wave_ax.set_xticks([])
            fft_ax.set_xticks([])

    plt.suptitle(name)
    plt.savefig(f'figures/{name}.png')
    plt.show()

first_half, second_half, heartrate_min, heartrate_max = split_data(patient_names[0])

first_half_dl, first_half_test = make_GAN_test_set(first_half)
second_half_dl, second_half_test = make_GAN_test_set(second_half)


wgan_gp_first = train_store_gan(first_half_dl, 'first_half', just_load=True, epochs=5000)
wgan_gp_second = train_store_gan(second_half_dl, 'second_half', just_load=True, epochs=5000)
synth_first = get_synthetic_outputs(wgan_gp_first, iterations=len(first_half_test))
synth_second = get_synthetic_outputs(wgan_gp_second, iterations=int(len(second_half_test)/2))
final_plot([synth_first, synth_second], [first_half_test,second_half_test], data_max=heartrate_max, data_min=heartrate_min, name='Patient 06-31-24 R-R First Second Half')
