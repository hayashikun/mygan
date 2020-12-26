import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn
from torchvision import datasets, transforms, utils as vutils

from mygan import TmpFilePath
from mygan.mnist.model import Discriminator, Generator

BATCH_SIZE = 300
LATENT_VECTOR_SIZE = 64

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataloader():
    mnist = datasets.MNIST(root=os.path.join(TmpFilePath, "mnist"),
                           train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5,), std=(0.5,))
                           ]),
                           download=True)
    return data_utils.DataLoader(dataset=mnist,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)


def train(n_epochs):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_VECTOR_SIZE, device=_device)

    net_d = Discriminator().to(_device)
    net_g = Generator(LATENT_VECTOR_SIZE).to(_device)
    d_optimizer = torch.optim.Adam(net_d.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(net_g.parameters(), lr=0.0002)

    g_losses = np.empty(shape=(n_epochs,))
    d_losses = np.empty(shape=(n_epochs,))

    dataloader = _dataloader()

    for epoch in range(n_epochs):
        d_total_loss = 0
        g_total_loss = 0
        for i, (images, _) in enumerate(dataloader):
            # Discriminator
            d_optimizer.zero_grad()
            images = images.view(BATCH_SIZE, -1).to(_device)
            real_labels = torch.ones(BATCH_SIZE, 1).to(_device)
            fake_labels = torch.zeros(BATCH_SIZE, 1).to(_device)
            outputs = net_d(images)
            d_loss_real = criterion(outputs, real_labels)
            noise = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, device=_device)
            fake_images = net_g(noise)
            outputs = net_d(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Generator
            g_optimizer.zero_grad()
            noise = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, device=_device)
            fake_images = net_g(noise)
            outputs = net_d(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            d_total_loss += d_loss.item()
            g_total_loss += g_loss.item()

        g_losses[epoch] = g_total_loss / len(dataloader)
        d_losses[epoch] = d_total_loss / len(dataloader)

        print(
            f'[{epoch}/{n_epochs}]'
            f'\tDiscriminator Loss: {d_losses[epoch]:.4f}'
            f'\tGenerator Loss: {g_losses[epoch]:.4f}'
        )

        with torch.no_grad():
            fake_images = net_g(fixed_noise).detach().cpu()
            fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

        img = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=True)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(np.transpose(img, (1, 2, 0)))
        fig.tight_layout()
        fig.savefig(os.path.join(TmpFilePath, f"mnist_gen_{epoch}.png"),
                    bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

    fig, ax = plt.subplots()
    ax.plot(g_losses, label="Generator")
    ax.plot(d_losses, label="Discriminator")
    ax.set(xlim=(0, n_epochs), xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(TmpFilePath, f"mnist_loss.png"))
    plt.close()
