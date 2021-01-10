import fire

from mygan.mnist.trainer import Trainer


def train(epochs):
    trainer = Trainer()
    trainer.train(epochs)


if __name__ == "__main__":
    fire.Fire({
        "train": train
    })
