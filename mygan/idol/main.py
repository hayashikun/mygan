import fire

from mygan.idol.trainer import Trainer


def train(epochs, tag, data_path, output_path=None):
    trainer = Trainer(tag, data_path, output_path)
    trainer.train(epochs)


def export(to, tag, output_path=None):
    if to == "js":
        export_tfjs(tag, output_path)


def export_tfjs(tag, output_path=None):
    trainer = Trainer(tag, "", output_path)
    trainer.export_tfjs_generator()


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "export": export,
    })
