import os

import fire
import logging

from mygan import TmpFilePath, s3
from mygan.face import gan, capture
from mygan import kanji
from mygan.kanji import gan as kanji_gan

logging.basicConfig(level=logging.INFO)


def camera():
    capture.capture()


def sync():
    s3.sync(os.path.join(TmpFilePath, "face"), "mygan/face/")
    s3.sync(os.path.join(TmpFilePath, "model"), "mygan/face_model/")


def face_train(n_epochs=20):
    gan.train(n_epochs)


def make_kanji_images():
    kanji.make_kanji_images()


def kanji_train(n_epochs=20):
    kanji_gan.train(n_epochs)


if __name__ == "__main__":
    fire.Fire()
