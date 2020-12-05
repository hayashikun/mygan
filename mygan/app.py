import fire
from mygan import capture


def camera():
    capture.capture()


if __name__ == "__main__":
    fire.Fire()
