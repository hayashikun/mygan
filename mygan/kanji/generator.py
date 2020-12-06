from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_vector_size, image_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        print("Generator", self.main)

    def forward(self, input_vec):
        return self.main(input_vec)
