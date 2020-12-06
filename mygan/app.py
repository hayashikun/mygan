import os

import fire

from mygan import TmpFilePath, capture, s3, gan


def camera():
    capture.capture()


def sync_face():
    s3.sync(os.path.join(TmpFilePath, "face"), "mygan/face/")


def train(n_epochs=50):
    gan.train(n_epochs)


if __name__ == "__main__":
    fire.Fire()
