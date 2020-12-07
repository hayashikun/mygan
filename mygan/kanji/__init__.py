from PIL import Image, ImageDraw, ImageFont
import os
from mygan import PackageRoot, TmpFilePath

KANJI_IMG_PATH = os.path.join(TmpFilePath, "kanji")

if not os.path.exists(KANJI_IMG_PATH):
    os.makedirs(KANJI_IMG_PATH)


def make_kanji_images():
    with open(os.path.join(PackageRoot, "kanji.txt")) as fp:
        chars = fp.read().split()
    for i, c in enumerate(chars):
        img = Image.new("L", (32, 32), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 30)
        draw.text((1, 1), c, font=font)
        img.save(os.path.join(KANJI_IMG_PATH, f"{i}.png"))
