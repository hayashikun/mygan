import os

import fire

from mygan import TmpFilePath, capture, s3


def camera():
    capture.capture()


def sync_face():
    s3.sync(os.path.join(TmpFilePath, "face"), "mygan/face/")

if __name__ == "__main__":
    fire.Fire()
