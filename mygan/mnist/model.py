from torch import nn

HIDDEN_SIZE = 256
IMAGE_SIZE = 28 ** 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(HIDDEN_SIZE, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_size, HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
                                 nn.Tanh())

    def forward(self, x):
        return self.net(x)
