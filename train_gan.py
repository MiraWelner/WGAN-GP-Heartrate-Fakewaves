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

epochs = 100
itt = 100


#first it makes sure that the machine you are running on has a GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("No GPU found")
    sys.exit()
mode = 'ordered'

square_data = np.loadtxt(f'processed_data/square_wave_{mode}.csv', delimiter=',')
triangle_data = np.loadtxt(f'processed_data/triangle_wave_{mode}.csv', delimiter=',')

#train/test split
train_square = np.expand_dims(square_data[square_data.shape[0]//3:],1)
test_square = np.expand_dims(square_data[:square_data.shape[0]//3],1)
train_triangle = np.expand_dims(triangle_data[triangle_data.shape[0]//3:],1)
test_triangle = np.expand_dims(triangle_data[:triangle_data.shape[0]//3],1)

#generate pytorch dataloaders
def generate_dataloader(data):
    torch_data = torch.from_numpy(data.astype(np.float32))
    dataloader = DataLoader(TensorDataset(torch_data), batch_size=256, shuffle=False, num_workers=6, drop_last=True)
    return dataloader
train_square_dl = generate_dataloader(train_square)
test_square_dl = generate_dataloader(test_square)
train_triangle_dl = generate_dataloader(train_triangle)
test_triangle_dl = generate_dataloader(test_triangle)

#train and store models
#"""
gen_square, _ = train_wgan(train_square_dl,test_square_dl, epochs=epochs)
torch.save(gen_square.state_dict(), f"models/generator_square_{mode}_{epochs}.pth")
gen_triangle, _ = train_wgan(train_triangle_dl,test_triangle_dl, epochs=epochs)
torch.save(gen_triangle.state_dict(), f"models/generator_triangle_{mode}_{epochs}.pth")
#"""


gen_triangle, gen_square = Generator().cuda(), Generator().cuda()
gen_triangle.load_state_dict(torch.load(f"models/generator_triangle_{mode}_{epochs}.pth", weights_only=True))
gen_square.load_state_dict(torch.load(f"models/generator_square_{mode}_{epochs}.pth", weights_only=True))

#test models
square_gen = np.array([gen_square(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(itt)])
square_real = test_square.squeeze()

triangle_gen = np.array([gen_triangle(torch.randn(1, 100).cuda()).cpu().detach().numpy().squeeze() for _ in range(itt)])
triangle_real = test_triangle.squeeze()

_, axes = plt.subplots(2,1, figsize=(17,7))

#plot square wave data on upper plot:
confidence_interval = 1.96 * np.std(square_gen, axis=0)/np.sqrt(square_gen.shape[1])
axes[0].errorbar(range(square_gen.shape[1]), np.mean(square_gen, axis=0), yerr=confidence_interval,alpha=0.1, color='cornflowerblue')
axes[0].plot(np.mean(square_gen, axis=0), label="Generated Data — mean and 95% confidence", color='cornflowerblue')
confidence_interval = 1.96 * np.std(square_real, axis=0)/np.sqrt(square_real.shape[1])
axes[0].errorbar(range(square_real.shape[1]), np.mean(square_real, axis=0), yerr=confidence_interval,alpha=0.1, color='orange')
axes[0].plot(np.mean(square_real, axis=0), label="Test set — mean and 95% confidence", color='orange')
axes[0].legend()
axes[0].set_xlabel("Time (s)")
axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
square_gen_mean = np.mean(square_gen)
square_gen_median = np.median(square_gen)
square_real_mean = np.mean(square_real)
square_real_median = np.median(square_real)
axes[0].set_title(f"Comparison Between original Square Waves and WGAN-Generated Square Waves \n Generated Mean: {square_gen_mean:.3f}, Generated Median {square_gen_median:.3f}, Original Mean: {square_real_mean:.3f}, Original Median {square_real_median:.3f}")

#plot triangle wave data on lower plot
confidence_interval = 1.96 * np.std(triangle_gen, axis=0)/np.sqrt(triangle_gen.shape[1])
axes[1].errorbar(range(triangle_gen.shape[1]), np.mean(triangle_gen, axis=0), yerr=confidence_interval,alpha=0.1, color='cornflowerblue')
axes[1].plot(np.mean(triangle_gen, axis=0), label="Generated Data — mean and 95% confidence", color='cornflowerblue')
confidence_interval = 1.96 * np.std(triangle_real, axis=0)/np.sqrt(triangle_real.shape[1])
axes[1].errorbar(range(triangle_real.shape[1]), np.mean(triangle_real, axis=0), yerr=confidence_interval,alpha=0.1, color='orange')
axes[1].plot(np.mean(triangle_real, axis=0), label="Test set — mean and 95% confidence", color='orange')
axes[1].legend()
axes[1].set_xlabel("Time (s)")
axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
triangle_gen_mean = np.mean(triangle_gen)
triangle_gen_median = np.median(triangle_gen)
triangle_real_mean = np.mean(triangle_real)
triangle_real_median = np.median(triangle_real)
axes[1].set_title(f"Comparison Between original Triangle Waves and WGAN-Generated Triangle Waves \n Generated Mean: {triangle_gen_mean:.3f}, Generated Median {triangle_gen_median:.3f}, Original Mean: {triangle_real_mean:.3f}, Original Median {triangle_real_median:.3f}")


plt.tight_layout()
plt.savefig(f"figures/wgan_comparison/{mode}_{epochs}.png")
plt.show()

_, axes = plt.subplots(2,1, figsize=(17,7))
axes[0].plot(square_gen[0], color='cornflowerblue')
axes[0].set_title("Example of Generated Square Output")
axes[0].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
axes[0].set_xlabel("Time (s)")

axes[1].plot(triangle_gen[0], color='cornflowerblue')
axes[1].set_title("Example of Generated Triangle Wave")
axes[1].set_xticks(np.arange(0, 3501, 500), [str(i) for i in range(8)])
axes[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"figures/samples/{mode}.png")
plt.show()
