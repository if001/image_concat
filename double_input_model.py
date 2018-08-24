import os
import numpy as np
from PIL import Image
from model_mod.model_opt_two_input_model import ModelOpt
from data_opt_mod.data_opt import DataOpt
WEIGHT_DIR = "./model_mod/weight"
DATA_DIR = "./data_opt_mod/get_pokemon_img/img"


def train():
    train_data1, teach_data1 = DataOpt.make_train_data_random_choice(
        DATA_DIR, 6000)
    train_data2, teach_data2 = DataOpt.make_train_data_random_choice(
        DATA_DIR, 6000)

    encoder_weight_file = os.path.join(WEIGHT_DIR, "encoder_double.hdf5")
    decoder_weight_file = os.path.join(WEIGHT_DIR, "decoder_double.hdf5")
    model_opt = ModelOpt(encoder_weight_file,
                         decoder_weight_file,
                         model_init=True)
    model_opt.train([train_data1, train_data2], [teach_data1, teach_data2])
    model_opt.save_model()


def predict():
    test_data1, _ = DataOpt.make_train_data_random_choice(DATA_DIR, 1)
    test_data2, _ = DataOpt.make_train_data_random_choice(DATA_DIR, 1)

    encoder_weight_file = os.path.join(WEIGHT_DIR, "encoder_double.hdf5")
    decoder_weight_file = os.path.join(WEIGHT_DIR, "decoder_double.hdf5")
    model_opt = ModelOpt(encoder_weight_file,
                         decoder_weight_file,
                         model_init=False)

    res1, res2 = model_opt.predict([test_data1, test_data2])

    res1 = (res1 * 255.)
    pil_img = Image.fromarray(np.uint8(res1[0])).save(
        'create_img/double_img_res1.png')

    res2 = (res2 * 255.)
    pil_img = Image.fromarray(np.uint8(res2[0])).save(
        'create_img/double_img_res2.png')


def main():
    # train()
    predict()


if __name__ == "__main__":
    main()
