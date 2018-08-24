

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import unicodedata
import os
import numpy as np
import sys
sys.path.append("../")

DIR_PATH = "./get_pokemon_img/img/"
IGNORE_FILE = [".DS_Store", "color.csv", "ignore", ".gitkeep"]


def ignore_file(file_list, remove_files):
    for remove_file in remove_files:
        while (remove_file in file_list):
            file_list.remove(remove_file)
    return file_list[::]


def set_prefix(fname, prefix):
    fname, ext = os.path.splitext(fname)
    new_fname = fname + "_" + prefix + ext
    return new_fname


def save_image(img_fname, prefix, processing=None, pos=(0, 0), rad=None):
    image = Image.open(os.path.join(DIR_PATH, img_fname))

    if processing is not None:
        image = processing(image)
    if rad is not None:
        image = image.rotate(rad)

    new_img_fname = set_prefix(img_fname, prefix)

    print("save :", os.path.join(DIR_PATH, new_img_fname))
    image.save(os.path.join(DIR_PATH, new_img_fname), 'PNG')


def contrast_50(img):
    img = ImageEnhance.Contrast(img)
    img = img.enhance(0.5)
    return img


def sharpness_2(img):
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(2.0)  # シャープ画像
    return img


def sharpness_0(img):
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(0.0)  # ボケ画像
    return img


def gaussian_bluer(img):
    img = img.filter(ImageFilter.GaussianBlur(3.0))
    return img


def erosion(img):  # 縮小
    img = img.filter(ImageFilter.MinFilter())
    return img


def dilation(img):  # 膨張
    img = img.filter(ImageFilter.MaxFilter())
    return img


def clean(prefix_list):
    img_dirs = ignore_file(os.listdir(DIR_PATH), IGNORE_FILE)
    for img_dir in img_dirs:
        img_fname_list = ignore_file(os.listdir(
            DIR_PATH + img_dir), IGNORE_FILE)
        for img_fname in img_fname_list:
            for prefix in prefix_list:
                new_img_fname = set_prefix(img_fname, prefix)
                file_path = os.path.join(DIR_PATH, img_dir, new_img_fname)
                if os.path.isfile(file_path):
                    print("remove ", file_path)
                    os.remove(file_path)


def main():
    if sys.argv[-1] == "--clean":
        prefix_list = [
            "mirror",
            "mirror_mirror",
            "mirror_flip",
            "flip",
            "mirror_flip_mirror",
            "mirror_mirror_mirro",
            "mirror_mirror_flip",
            "mirror_mirror_flip_mirror",
            "mirror_mirror_mirror_mirror",
            "mirror_mirror_rad180",
            "mirror_mirror_rad180_mirror",
            "mirror_rad180",
            "mirror_rad180_mirror",
            "rad180",
            "rad180_mirror"
        ]
        clean(prefix_list)
        exit(0)

    img_fname_list = ignore_file(os.listdir(DIR_PATH), IGNORE_FILE)

    for img_fname in img_fname_list:
        save_image(img_fname, prefix="mirror",
                   processing=ImageOps.mirror)
        save_image(img_fname, prefix="rad90",
                   rad=90)
        save_image(img_fname, prefix="rad270",
                   rad=270)
        save_image(img_fname, prefix="flip",
                   processing=ImageOps.flip)


if __name__ == '__main__':
    main()
