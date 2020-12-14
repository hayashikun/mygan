import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, utils as vutils

from mygan import TmpFilePath
# from mygan.kanji.model import Discriminator, Generator, IMAGE_SIZE
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


def train(n_epochs):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=_device)
    real_label = 1.
    fake_label = 0.

    net_g = Generator(LATENT_VECTOR_SIZE).to(_device)
    net_d = Discriminator().to(_device)
    if os.path.exists(MODEL_G_PATH):
        net_g.load_state_dict(torch.load(MODEL_G_PATH))
    if os.path.exists(MODEL_D_PATH):
        net_d.load_state_dict(torch.load(MODEL_D_PATH))

    optimizer_g = optim.Adam(net_g.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler_g = StepLR(optimizer_g, step_size=1000, gamma=0.8)
    scheduler_d = StepLR(optimizer_d, step_size=1000, gamma=0.8)

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

                if (epoch + 1) % 1000 == 0:
                    torch.save(net_g.state_dict(), MODEL_G_PATH.format(epoch))
                    torch.save(net_d.state_dict(), MODEL_D_PATH.format(epoch))

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

                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(g_losses[:epoch + 1], color="red", label="G")
                ax2.plot(d_losses[:epoch + 1], color="blue", label="D")
                ax1.set(xlim=(0, epoch + 1), ylim=(0, g_losses[:epoch + 1].mean() * 2), xlabel="Epoch",
                        ylabel="G Loss")
                ax2.set(ylim=(0, d_losses[:epoch + 1].mean() * 2), ylabel="D Loss")

                for ax, c in zip([ax1, ax2], ["red", "blue"]):
                    ax.set_ylabel(ax.get_ylabel(), color=c)
                    ax.tick_params(axis="y", colors=c)
                    ax.spines[ax.yaxis.get_ticks_position()].set_color(c)

                fig.tight_layout()
                fig.savefig(os.path.join(TmpFilePath, f"loss.png"))
                plt.close()

                scheduler_d.step()
                scheduler_g.step()
