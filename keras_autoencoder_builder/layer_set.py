from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import concatenate
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers.normalization import BatchNormalization
import sys
sys.path.append("../")
import const


class LayerSet():
    @classmethod
    def encoder_input(cls):
        return Input(shape=(const.IMG_SIZE, const.IMG_SIZE, 3))

    @classmethod
    def encoder_layer(cls, input_layer):
        x = Conv2D(64, (3, 3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        return encoded

    @classmethod
    def decoder_input(cls):
        return Input(shape=(8, 8, 32))

    @classmethod
    def decoder_layer(cls, input_layer):
        x = Conv2D(32, (3, 3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        # decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        return decoded
