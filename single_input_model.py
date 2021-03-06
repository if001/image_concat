import os
import numpy as np
from PIL import Image
from model_mod.model_opt import ModelOpt
from data_opt_mod.data_opt import DataOpt
WEIGHT_DIR = "./model_mod/weight"
DATA_DIR = "./data_opt_mod/get_pokemon_img/img"
DATA_SIZE = 12000

TRAIN_DATA_SAVE_FILE = "./data_opt_mod/train.npz"


def train():
    encoder_weight_file = os.path.join(WEIGHT_DIR, "encoder.hdf5")
    decoder_weight_file = os.path.join(WEIGHT_DIR, "decoder.hdf5")
    model_opt = ModelOpt(encoder_weight_file,
                         decoder_weight_file,
                         model_init=True)

    # train_data, teach_data = DataOpt.make_train_data_random_choice(
    #     DATA_DIR, DATA_SIZE)
    # DataOpt.save_data_set(train_data, teach_data,
    #                       TRAIN_DATA_SAVE_FILE)

    train_data, teach_data = DataOpt.load_data_set(TRAIN_DATA_SAVE_FILE)

    model_opt.train(train_data, teach_data)
    model_opt.save_model()

    # for _ in range(20):
    #     train_data, teach_data = DataOpt.make_train_data_random_choice(
    #         DATA_DIR, DATA_SIZE)
    #     model_opt.train(train_data, teach_data)
    #     model_opt.save_model()


def predict():
    test_data, _ = DataOpt.make_train_data_random_choice(DATA_DIR, 1)

    encoder_weight_file = os.path.join(WEIGHT_DIR, "encoder.hdf5")
    decoder_weight_file = os.path.join(WEIGHT_DIR, "decoder.hdf5")
    model_opt = ModelOpt(encoder_weight_file,
                         decoder_weight_file,
                         model_init=False)

    result = model_opt.predict(test_data)
    print(result)
    result = (result * 255.)
    pil_img = Image.fromarray(np.uint8(result[0])).save(
        'create_img/single_img.png')


def main():
    train()
    # predict()


if __name__ == "__main__":
    main()
