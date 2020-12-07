import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torchvision import datasets, transforms, utils as vutils

from mygan import TmpFilePath
from mygan.kanji.discriminator import Discriminator
from mygan.kanji.generator import Generator

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
N_CHANNEL = 1
KERNEL_SIZE = 2
LATENT_VECTOR_SIZE = 100
BATCH_SIZE = 200
MODEL_PATH = os.path.join(TmpFilePath, "kanji_model")
MODEL_G_PATH = os.path.join(MODEL_PATH, "generator.pt")
MODEL_D_PATH = os.path.join(MODEL_PATH, "discriminator.pt")

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
                                       transforms.Normalize(0.5, 0.5),
                                   ]),
                                   is_valid_file=is_valid_file)
    return data_utils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)


def train(n_epochs):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=_device)
    real_label = 1.
    fake_label = 0.
    lr = 0.0002  # Learning rate
    beta1 = 0.5  # Beta1 hyper parameter

    net_g = Generator(LATENT_VECTOR_SIZE, IMAGE_SIZE).to(_device)
    net_d = Discriminator(IMAGE_SIZE).to(_device)
    if os.path.exists(MODEL_G_PATH):
        net_g.load_state_dict(torch.load(MODEL_G_PATH))
    if os.path.exists(MODEL_D_PATH):
        net_d.load_state_dict(torch.load(MODEL_D_PATH))

    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

    g_losses = np.empty(shape=(n_epochs,))
    d_losses = np.empty(shape=(n_epochs,))

    dataloader = _dataloader()

    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            net_d.zero_grad()
            real_cpu = data[0].to(_device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=_device)
            output = net_d(real_cpu).view(-1)
            err_d_real = criterion(output, label)
            err_d_real.backward()
            d_x = output.mean().item()

            # create fake images using generator
            noise = torch.randn(b_size, LATENT_VECTOR_SIZE, 1, 1, device=_device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach()).view(-1)
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            d_g_z1 = output.mean().item()
            err_d = err_d_real + err_d_fake
            # update Discriminator
            optimizer_d.step()

            net_g.zero_grad()
            label.fill_(real_label)
            output = net_d(fake).view(-1)
            err_g = criterion(output, label)
            err_g.backward()
            d_g_z2 = output.mean().item()
            # update Generator
            optimizer_g.step()

            if i == len(dataloader) - 1:
                g_losses[epoch] = err_g.item()
                d_losses[epoch] = err_d.item()

                torch.save(net_g.state_dict(), MODEL_G_PATH)
                torch.save(net_d.state_dict(), MODEL_D_PATH)

                print(f'[{epoch}/{n_epochs}]'
                      f'\tDiscriminator Loss: {err_d.item():.4f}'
                      f'\tGenerator Loss: {err_g.item():.4f}'
                      f'\tD(x): {d_x:.4f}'
                      f'\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}')

                with torch.no_grad():
                    fake = net_g(fixed_noise).detach().cpu()

                img = vutils.make_grid(fake, nrow=8, padding=2, normalize=True)
                fig, ax = plt.subplots()
                ax.set_axis_off()
                ax.imshow(np.transpose(img, (1, 2, 0)))
                fig.tight_layout()
                fig.savefig(os.path.join(TmpFilePath, f"gen_{epoch}.png"),
                            bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close()

    fig, ax = plt.subplots()
    ax.plot(g_losses, label="G")
    ax.plot(d_losses, label="D")
    ax.set(xlim=(0, n_epochs), xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(TmpFilePath, f"loss.png"))
