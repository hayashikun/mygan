import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torchvision import datasets, transforms, utils as vutils

from mygan import TmpFilePath
from mygan.kanji.model_32 import Discriminator, Generator, IMAGE_SIZE

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CHANNEL = 1
KERNEL_SIZE = 2
LATENT_VECTOR_SIZE = 100
BATCH_SIZE = 200
MODEL_PATH = os.path.join(TmpFilePath, "kanji_model")
MODEL_G_PATH = os.path.join(MODEL_PATH, "generator_{}.pt")
MODEL_D_PATH = os.path.join(MODEL_PATH, "discriminator_{}.pt")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def _dataloader():
    def is_valid_file(x):
        return "tmp/kanji/" in x

    workers = 2
    dataset = datasets.ImageFolder(root=TmpFilePath,
                                   transform=transforms.Compose([
                                       transforms.Grayscale(),
                                       transforms.Resize(IMAGE_SIZE),
                                       transforms.ToTensor(),
                                   ]),
                                   is_valid_file=is_valid_file)
    return data_utils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)


def train(n_epochs, save_model_interval, init_epoch=0):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=_device)

    net_g = Generator(LATENT_VECTOR_SIZE).to(_device)
    net_d = Discriminator().to(_device)
    if init_epoch > 0:
        net_g.load_state_dict(torch.load(MODEL_G_PATH.format(init_epoch - 1)))
        net_d.load_state_dict(torch.load(MODEL_D_PATH.format(init_epoch - 1)))

    g_optimizer = optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

    g_losses = np.empty(shape=(n_epochs - init_epoch,))
    d_losses = np.empty(shape=(n_epochs - init_epoch,))

    dataloader = _dataloader()

    for epoch in range(init_epoch, n_epochs):
        d_total_loss = 0
        g_total_loss = 0
        for i, (images, _) in enumerate(dataloader, 0):
            images = images.to(_device)
            b_size = images.size(0)
            real_labels = torch.ones(b_size, ).to(_device)
            fake_labels = torch.zeros(b_size, ).to(_device)

            # Discriminator
            d_optimizer.zero_grad()
            outputs = net_d(images).view(-1)
            d_loss_real = criterion(outputs, real_labels)
            noise = torch.randn(b_size, LATENT_VECTOR_SIZE, 1, 1, device=_device)
            fake_images = net_g(noise)
            outputs = net_d(fake_images.detach()).view(-1)
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Generator
            g_optimizer.zero_grad()
            outputs = net_d(fake_images).view(-1)
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

        img = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=True)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(np.transpose(img, (1, 2, 0)))
        fig.tight_layout()
        fig.savefig(os.path.join(TmpFilePath, f"kanji_gen_{epoch}.png"),
                    bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        if (epoch + 1) % save_model_interval == 0:
            torch.save(net_g.state_dict(), MODEL_G_PATH.format(epoch))
            torch.save(net_d.state_dict(), MODEL_D_PATH.format(epoch))

            fig, ax = plt.subplots()
            ax.plot(g_losses[:epoch + 1], label="Generator")
            ax.plot(d_losses[:epoch + 1], label="Discriminator")
            ax.set(xlim=(init_epoch, epoch + 1), xlabel="Epoch", ylabel="Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(TmpFilePath, f"kanji_loss.png"))
            plt.close()
