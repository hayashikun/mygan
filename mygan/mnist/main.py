import fire

from mygan.mnist.trainer import train

if __name__ == "__main__":
    fire.Fire({
        "train": train
    })
