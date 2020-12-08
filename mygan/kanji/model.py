from torch import nn

IMAGE_SIZE = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, IMAGE_SIZE, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_SIZE, IMAGE_SIZE * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_SIZE * 2, IMAGE_SIZE * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_SIZE * 4, IMAGE_SIZE * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_SIZE * 8, 1, 1, 4, 0, bias=False),
            nn.Sigmoid()
        )
        print("Discriminator", self.main)

    def forward(self, input_vec):
        return self.main(input_vec)


class Generator(nn.Module):
    def __init__(self, latent_vector_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, IMAGE_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_SIZE * 8, IMAGE_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_SIZE * 4, IMAGE_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_SIZE * 2, IMAGE_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_SIZE, 1, 2, 2, 0, bias=False),
            nn.Tanh()
        )
        print("Generator", self.main)

    def forward(self, input_vec):
        return self.main(input_vec)
