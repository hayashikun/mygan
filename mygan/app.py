import logging
import os

import fire

from mygan import TmpFilePath, s3
from mygan import kanji
from mygan.face import gan, capture
from mygan.kanji import gan as kanji_gan
from mygan.mnist import gan as mnist_gan

logging.basicConfig(level=logging.INFO)


def camera():
    capture.capture()


def sync():
    s3.sync(os.path.join(TmpFilePath, "face"), "mygan/face/")
    s3.sync(os.path.join(TmpFilePath, "model"), "mygan/face_model/")


def face_train(n_epochs=500):
    gan.train(n_epochs)


def make_kanji_images():
    kanji.make_kanji_images()


def kanji_train(n_epochs=500, save_model_interval=100, init_epoch=0):
    kanji_gan.train(n_epochs, save_model_interval, init_epoch)


def mnist_train(n_epochs=500):
    mnist_gan.train(n_epochs)


if __name__ == "__main__":
    fire.Fire()
