import os

from PIL import Image, ImageDraw, ImageFont

from mygan import PackageRoot, TmpFilePath

KANJI_IMG_PATH = os.path.join(TmpFilePath, "kanji")

if not os.path.exists(KANJI_IMG_PATH):
    os.makedirs(KANJI_IMG_PATH)


def font_location():
    for loc in [
        "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",  # for Mac
        "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",  # for Linux
        "/mnt/c/Windows/Fonts/yumin.ttf"  # for WSL
    ]:
        if os.path.exists(loc):
            return loc
    raise Exception("Font not found.")


def make_kanji_images():
    with open(os.path.join(PackageRoot, "kanji.txt")) as fp:
        chars = fp.read().split()
    for i, c in enumerate(chars):
        img = Image.new("L", (64, 64), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_location(), 60)
        draw.text((2, 2), c, font=font)
        img.save(os.path.join(KANJI_IMG_PATH, f"{i}.png"))
