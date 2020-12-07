from PIL import Image, ImageDraw, ImageFont
import os
import platform
from mygan import PackageRoot, TmpFilePath

KANJI_IMG_PATH = os.path.join(TmpFilePath, "kanji")

if not os.path.exists(KANJI_IMG_PATH):
    os.makedirs(KANJI_IMG_PATH)


def font_location():
    pf = platform.system()
    if pf == 'Darwin':
        return "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc"
    else:
        return "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf"


def make_kanji_images():
    with open(os.path.join(PackageRoot, "kanji.txt")) as fp:
        chars = fp.read().split()
    for i, c in enumerate(chars):
        img = Image.new("L", (64, 64), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_location(), 60)
        draw.text((2, 2), c, font=font)
        img.save(os.path.join(KANJI_IMG_PATH, f"{i}.png"))
