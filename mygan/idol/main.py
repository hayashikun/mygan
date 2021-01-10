import fire

from mygan.idol.trainer import Trainer


def train(epochs, tag, data_path):
    trainer = Trainer(tag, data_path)
    trainer.train(epochs)


if __name__ == "__main__":
    fire.Fire({
        "train": train
    })
