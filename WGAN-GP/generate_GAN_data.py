from wgan import Generator
import matplotlib.pyplot as plt
import torch

seconds = 7
hz=500
itt = 10
channels = ['ecg0_channel0']
latent_dim=100

generator = Generator(signal_length=seconds*hz, latent_dim=latent_dim, channels=len(channels)).cuda()
generator.load_state_dict(torch.load(f"../models/generator_weights_{'_'.join(channels)}.pth", weights_only=True))

z = torch.randn(itt, latent_dim).cuda()

fake_signal = generator(z).cpu().detach().numpy().T
_, axes = plt.subplots(len(channels)*itt,1, figsize=(17,len(channels)*itt*5))
for i in range(len(channels)):
    for j in range(itt):
        axes[i*itt + j].plot(fake_signal[:,i,j])
        axes[i*itt + j].set_yticks([])
        axes[i*itt + j].set_ylim(-1,1)
plt.tight_layout()
plt.savefig("../figures/generate_signals.png")
plt.show()
