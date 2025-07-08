import numpy as np
import torch
from torch.nn import Linear, LeakyReLU, Sequential, Tanh, Module
from torch.optim.rmsprop import RMSprop #RMSprop rather than ADAM bc https://dl.acm.org/doi/pdf/10.5555/3305381.3305404
from tqdm import trange

class Generator(Module):
    """
    The Generator takes in noise of length latent_dim and outputs
    signals similar to the data it was trained on. The training data
    can be any number of channels. It is expected that these channels
    of data are related to eachother (ECG signals taken
    from the same patient at the same time)
    """
    def __init__(self, latent_dim=100, signal_length=3500, channels=1):
        super().__init__()
        self.signal_length = signal_length
        self.channels = channels

        self.model = Sequential(
            Linear(latent_dim, 256),
            LeakyReLU(),
            Linear(256, 512),
            LeakyReLU(),
            Linear(512, 1024),
            LeakyReLU(),
            Linear(1024, signal_length*channels),
            Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(z.size(0), self.channels, self.signal_length)
        return x

class Critic(Module):
    """
    The GAN is a Wasserstein Gan with Gradient Penalty so
    it outputs a raw score which then has gradient penalty
    applied to it
    """
    def __init__(self, signal_length, channels):
        super().__init__()
        in_dim = channels * signal_length

        self.model = Sequential(
            Linear(in_dim, 512),
            LeakyReLU(0.2),
            Linear(512, 256),
            LeakyReLU(0.2),
            Linear(256, 128),
            LeakyReLU(0.2),
            Linear(128, 1)  # raw score
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)




def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.shape).cuda()
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan(dataloader, epochs, latent_dim=100, signal_length=3500, batch_size=256, channels=1):
    d_losses = []
    g_losses = []
    generator = Generator(latent_dim, signal_length, channels).cuda()
    discriminator = Critic(signal_length, channels).cuda()

    lr = 1e-4
    optimizer_G = RMSprop(generator.parameters(), lr=lr)
    optimizer_D = RMSprop(discriminator.parameters(), lr=lr)

    lambda_gp = 10
    n_critic = 5

    for epoch in trange(epochs):
        d_loss_batch = []
        g_loss_batch = []
        for i, (real_batch,) in enumerate(dataloader):
            real_batch = real_batch.cuda()
            for _ in range(n_critic):
                z = torch.randn(real_batch.size(0), latent_dim).cuda()
                fake_batch = generator(z)
                d_real = discriminator(real_batch).mean()
                d_fake = discriminator(fake_batch).mean()

                gradient_penalty = compute_gradient_penalty(discriminator, real_batch, fake_batch)
                d_loss = -d_real + d_fake + lambda_gp * gradient_penalty

                optimizer_D.zero_grad()
                d_loss.backward()
                d_loss_batch.append(d_loss.cpu().detach().numpy().item())
                optimizer_D.step()

            z = torch.randn(real_batch.size(0), latent_dim).cuda()
            gen_signals = generator(z)
            g_loss = -discriminator(gen_signals).mean()
            g_loss_batch.append(g_loss.cpu().detach().numpy().item())

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        d_losses.append(np.mean(np.array(d_loss_batch)))
        g_losses.append(np.mean(np.array(g_loss_batch)))
        torch.cuda.empty_cache()


    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,5),)
    plt.suptitle("Training Error for the WGAN-GP")
    axes[0].plot(g_losses)
    axes[0].set_title("Generator Loss")
    axes[0].set_xticks(np.arange(0, len(g_losses) + 1, len(g_losses)//5))
    axes[0].set_xlabel("Epochs")

    axes[1].plot(d_losses)
    axes[1].set_title("critic loss")
    axes[1].set_xticks(np.arange(0, len(d_losses) + 1, len(d_losses)//5))
    axes[1].set_xlabel("Epochs")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../figures/training_error_{channels}_channels_{epochs}_epochs.png")
    """
    return generator, discriminator
