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
from random import randrange


def increment(mode, peak_distance, ordered_it):
    """
    When developing the waveform to run through the WGAN-GP, this
    function increments the distance between each waveform depending on
    whether you want an evenly spaced wavefunction or a randomly spaced one
    or a patterned spaced one
    """
    ordered = [5,1,2,7,3,4,10,1,6,5]
    match mode:
        case 'even':
            return peak_distance
        case 'random':
            return int(randrange(10)*peak_distance/10)
        case 'ordered':
            i =  ordered[ordered_it]*peak_distance/10
            if ordered_it+1 == len(ordered):
                ordered_it = 0
            else:
                ordered_it += 1
            return int(i)
    return 0

def create_square_wave(mode, peak_width, peak_distance, peak_height):
    ordered_it = 0
    high_if_mixed = 0 #if the peak_height is 'mixed' this counter determines high or low
    x = np.full(3500, -1.0)
    n = 0
    while n < 3500:
        n += increment(mode, peak_distance, ordered_it)
        if ordered_it < 9:
            ordered_it += 1
        else:
            ordered_it= 0
        if peak_height == 'high' or (peak_height == 'mixed' and high_if_mixed == 1):
            x[n:int(n+peak_width)] = 1
            high_if_mixed = 0
        else:
            x[n:int(n+peak_width)] = 0.5
            high_if_mixed = 1
        n += peak_width
    return x


def create_triangle_wave(mode, peak_width, peak_distance, peak_height):
    ordered_it = 0
    high_if_mixed = 0 #if the peak_height is 'mixed' this counter determines high or low
    x = np.full(3500, -1.0)
    n = 0
    while n + peak_width < 3500:
        n += increment(mode, peak_distance, ordered_it)
        if ordered_it < 9:
            ordered_it += 1
        else:
            ordered_it= 0
        if n + peak_width/2 >= 3500:
            break
        if peak_height == 'high' or (peak_height == 'mixed' and high_if_mixed == 1):
            up_slope = np.linspace(-1, 1, int(peak_width/2))
            down_slope = np.linspace(1, -1, int(peak_width/2))
            high_if_mixed = 0
        else:
            up_slope = np.linspace(-1, 0.5, int(peak_width/2))
            down_slope = np.linspace(0.5, -1, int(peak_width/2))
            high_if_mixed = 1

        x[n:int(n+peak_width/2)] = up_slope
        n += int(peak_width/2)

        if n + peak_width/2 >= 3500:
            break
        x[n:int(n+peak_width/2)] = down_slope
        n += int(peak_width/2)
    return x

def make_train_test_dataloaders(square_data, triangle_data):
    def generate_dataloader(data):
        torch_data = torch.from_numpy(data.astype(np.float32))
        dataloader = DataLoader(TensorDataset(torch_data), batch_size=256*2, shuffle=False, num_workers=6, drop_last=True)
        return dataloader

    #train/test split
    train_square = np.expand_dims(square_data[square_data.shape[0]//3:],1)
    test_square = np.expand_dims(square_data[:square_data.shape[0]//3],1)
    train_triangle = np.expand_dims(triangle_data[triangle_data.shape[0]//3:],1)
    test_triangle = np.expand_dims(triangle_data[:triangle_data.shape[0]//3],1)

    train_square_dl = generate_dataloader(train_square)
    train_triangle_dl = generate_dataloader(train_triangle)

    return train_square_dl, test_square, train_triangle_dl, test_triangle

def train_store_gan(train_square, train_triangle, gen_name = '', epochs=100, just_load=False):
    if not just_load:
        gen_square, _ = train_wgan(train_square, epochs=epochs)
        torch.save(gen_square.state_dict(), f"models/generator_square_{gen_name}.pth")
        gen_triangle, _ = train_wgan(train_triangle, epochs=epochs)
        torch.save(gen_triangle.state_dict(), f"models/generator_triangle_{gen_name}.pth")

    gen_triangle, gen_square = Generator().cuda(), Generator().cuda()
    gen_triangle.load_state_dict(torch.load(f"models/generator_triangle_{gen_name}.pth", weights_only=True))
    gen_square.load_state_dict(torch.load(f"models/generator_square_{gen_name}.pth", weights_only=True))
    return gen_square, gen_triangle

def plot_mean_and_diff(square_real, square_gen, triangle_real, triangle_gen, filename='comparison', show=False):
    _, axes = plt.subplots(2,1, figsize=(17,7))

    mean_square_real = np.mean(square_real, axis=0)
    mean_square_gen = np.mean(square_gen, axis=0)
    mean_diff = np.mean(square_real - mean_square_gen, axis=0)
    lower_bound = mean_square_real - np.minimum(mean_diff, np.zeros(len(mean_square_real)))
    upper_bound = mean_square_real - np.maximum(mean_diff, np.zeros(len(mean_square_real)))

    axes[0].fill_between(range(triangle_gen.shape[1]), lower_bound, upper_bound, alpha=0.5, label="Mean Difference Between Real and Gen")
    axes[0].plot(mean_square_real, label="Mean Real Data", color='cornflowerblue')
    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[0].set_ylim(-1.2,1.2)
    square_gen_mean = np.mean(square_gen)
    square_gen_median = np.median(square_gen)
    square_gen_std = np.std(square_gen)
    square_real_mean = np.mean(square_real)
    square_real_median = np.median(square_real)
    square_real_std = np.std(square_real)
    axes[0].set_title(f"Comparison Between original Square Waves and WGAN-Generated Square Waves \n Generated Mean: {square_gen_mean:.3f}, Generated Median {square_gen_median:.3f}, Generated STD {square_gen_std:.3f}, Original Mean: {square_real_mean:.3f}, Original Median {square_real_median:.3f}, Original STD {square_real_std:.3f}")

    #plot triangle wave data on lower plot
    mean_triangle_real = np.mean(triangle_real, axis=0)
    mean_triangle_gen = np.mean(triangle_gen, axis=0)
    mean_diff = np.mean(triangle_real - mean_triangle_gen, axis=0)
    lower_bound = mean_triangle_real + np.minimum(mean_diff, np.zeros(len(mean_triangle_real)))
    upper_bound = mean_triangle_real + np.maximum(mean_diff, np.zeros(len(mean_triangle_real)))

    axes[1].fill_between(range(triangle_gen.shape[1]), lower_bound, upper_bound, alpha=0.5, label="Mean Difference Between Real and Gen")
    axes[1].plot(mean_triangle_real, label="Mean Real Data", color='cornflowerblue')
    axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[1].set_ylim(-1.2,1.2)
    triangle_gen_mean = np.mean(triangle_gen)
    triangle_gen_median = np.median(triangle_gen)
    triangle_gen_std = np.std(triangle_gen)
    triangle_real_mean = np.mean(triangle_real)
    triangle_real_median = np.median(triangle_real)
    triangle_real_std = np.std(triangle_real)
    axes[1].set_title(f"Comparison Between original Triangle Waves and WGAN-Generated Triangle Waves \n Generated Mean: {triangle_gen_mean:.3f}, Generated Median {triangle_gen_median:.3f}, Generated STD {triangle_gen_std:.3f}, Original Mean: {triangle_real_mean:.3f}, Original Median {triangle_real_median:.3f}, Original STD {triangle_real_std:.3f}")

    plt.tight_layout()
    plt.savefig(f"figures/wgan_comparison/{filename}.png")
    if show:
        plt.show()

def plot_examples(square_gen, triangle_gen, filename='examples', show=False):
    _, axes = plt.subplots(2,1, figsize=(17,7))
    axes[0].plot(square_gen[0], color='cornflowerblue')
    axes[0].set_title("Example of Generated Square Output")
    axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylim(-1.2,1.2)


    axes[1].plot(triangle_gen[0], color='cornflowerblue')
    axes[1].set_title("Example of Generated Triangle Wave")
    axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-1.2,1.2)
    plt.tight_layout()
    plt.savefig(f"figures/samples/{filename}.png")
    if show:
        plt.show()


#ensure that the computer has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()

for mode in ['even', 'ordered', 'random']:
    for peak_width in [6,20,50]:
        for peak_distance in [6,20,50]:
            for peak_height in ['low','high','mixed']:
                name = f"{mode}_{peak_width}_{peak_distance}_{peak_height}"
                square_data = np.array([create_square_wave(mode=mode,
                                                           peak_width=peak_width,
                                                           peak_distance=peak_distance,
                                                           peak_height=peak_height)
                                        for _ in range(1000)])

                triangle_data = np.array([create_triangle_wave(mode=mode,
                                                               peak_width=peak_width,
                                                               peak_distance=peak_distance,
                                                               peak_height=peak_height)
                                            for _ in range(1000)])
                train_square, test_square, train_triangle, test_triangle = make_train_test_dataloaders(square_data, triangle_data)
                gen_square, gen_triangle = train_store_gan(train_square, train_triangle, gen_name=name)

                square_gen = np.array([gen_square(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(500)])
                square_real = test_square.squeeze()

                triangle_gen = np.array([gen_triangle(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(500)])
                triangle_real = test_triangle.squeeze()

                plot_mean_and_diff(square_real, square_gen, triangle_real, triangle_gen, filename=name)
                plot_examples(square_gen, triangle_gen, filename=name)
