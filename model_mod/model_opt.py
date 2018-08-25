import keras
import numpy as np
import os


import sys
sys.path.append("../")
from keras_autoencoder_builder.build import AutoEncoderBuidler


BATCH_SIZE = 64
EPOCHS = 20
VERBOSE = 1
BASE_PATH = ""

ENCODER_WEIGHT_FILE = ""
DECODER_WEIGHT_FILE = ""


class ModelOpt():
    def __init__(self, encoder_weight_file, decoder_weight_file, model_init=True):
        self.encoder_weight_file = encoder_weight_file
        self.decoder_weight_file = decoder_weight_file

        if model_init:
            self.__builder = AutoEncoderBuidler()
        else:
            self.__builder = AutoEncoderBuidler(
                encoder_weight_file=encoder_weight_file,
                decoder_weight_file=decoder_weight_file)

        self.cbs = ModelOpt.set_callbacks("auto_encoder_model")

    def train(self, train, teach):
        history = self.__builder.auto_encoder.fit(train, teach,
                                                  batch_size=BATCH_SIZE,
                                                  epochs=EPOCHS,
                                                  verbose=VERBOSE,
                                                  validation_split=0.1,
                                                  callbacks=self.cbs)
        return history

    def predict(self, train):
        score = self.__builder.auto_encoder.predict(train)
        return score

    @classmethod
    def set_callbacks(cls, prefix):
        callbacks = []
        # fpath = os.path.join(
        #     BASE_PATH, prefix + '_weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
        # callbacks.append(keras.callbacks.ModelCheckpoint(
        #     filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto'))

        callbacks.append(keras.callbacks.TensorBoard(log_dir="model_mod/tflog", histogram_freq=1))

        return callbacks

    def save_model(self):
        self.__builder.save_encoder(self.encoder_weight_file)
        self.__builder.save_decoder(self.decoder_weight_file)
