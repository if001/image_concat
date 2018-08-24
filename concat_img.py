from PIL import Image
import numpy as np

# import sys
# sys.path.append("../")
from data_opt_mod.data_opt import DataOpt
from keras_autoencoder_builder.build import AutoEncoderBuidler
from keras_autoencoder_builder_two_input.build import AutoEncoderBuidler as AutoEncoderBuidlerTwo

SAVE_DIR = "create_img/"


def predict(img_path1, img_path2):
    img1 = np.array([DataOpt.img_open(img_path1)])
    img2 = np.array([DataOpt.img_open(img_path2)])

    single = AutoEncoderBuidler(
        encoder_weight_file="./model_mod/weight/encoder.hdf5",
        decoder_weight_file="./model_mod/weight/decoder.hdf5"
    )
    double = AutoEncoderBuidlerTwo(
        encoder_weight_file="./model_mod/weight/encoder_double.hdf5",
        decoder_weight_file="./model_mod/weight/decoder_double.hdf5",
    )

    # 平均
    # encoded1 = single.encoder.predict(img1)
    # encoded2 = single.encoder.predict(img2)
    # encoded = (encoded1 + encoded2) / 2

    # double autoencoder
    encoded = double.encoder.predict([img1, img2])

    result = single.decoder.predict(encoded)
    result = (result * 255.)
    return result


def main():
    img_path1 = "./data_opt_mod/get_pokemon_img/img/004e03952b0150caf6f74db255dbfbdf.png"
    img_path2 = "./data_opt_mod/get_pokemon_img/img/1010fd06641d6ebeedd80a6efbaacdb7.png"
    result = predict(img_path1, img_path2)
    pil_img = Image.fromarray(np.uint8(result[0])).save(
        SAVE_DIR + 'concat_img.png')


if __name__ == "__main__":
    main()
