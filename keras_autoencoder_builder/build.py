from keras.models import Model
import os

import sys
sys.path.append("../")
# from layer_set import LayerSet
from keras_autoencoder_builder.layer_set import LayerSet

BASE_PATH = ""


class AutoEncoderBuidler():
    def __init__(self,
                 loss='binary_crossentropy',
                 opt='adam',
                 encoder_weight_file='',
                 decoder_weight_file=''
                 ):
        """
        loss: オブジェクト or String
        optimizer: オブジェクト or String
        encoder_weight_file_path: ファイル名が指定されていればモデルをロード
        decoder_weight_file_path: ファイル名が指定されていればモデルをロード
        """

        self.__loss = AutoEncoderBuidler.open_func(loss)
        self.__opt = AutoEncoderBuidler.open_func(opt)

        self.encoder = self.__model_builder(
            LayerSet.encoder_input(),
            [LayerSet.encoder_layer],
            encoder_weight_file
        )

        self.decoder = self.__model_builder(
            LayerSet.decoder_input(),
            [LayerSet.decoder_layer],
            decoder_weight_file
        )

        self.auto_encoder = self.__build_from_model(
            LayerSet.encoder_input(),
            [self.encoder, self.decoder]
        )

    @classmethod
    def open_func(cls, f):
        """
        optimizerやlossが、オブジェクト or Stringで飛んでくるため
        ここで開けて関数にする
        """
        return f if type(f) == str else f()

    def __model_builder(self, input_layer, layers, weight_file=''):
        if weight_file == '':
            __first_inp = input_layer
            for layer in layers:
                output_layer = layer(input_layer)
                input_layer = output_layer
            model = Model(__first_inp, output_layer)
            model.compile(optimizer=self.__opt, loss=self.__loss)
            model.summary()
        else:
            model = AutoEncoderBuidler.load_model(weight_file)
        return model

    def __build_from_model(self, input_layer, models):
        __concat_layers = []
        for model in models:
            __concat_layers += model.layers[1:]
        return self.__model_builder(input_layer, __concat_layers)

    def save_encoder(self, weight_file):
        AutoEncoderBuidler.__save_model(self.encoder, weight_file)

    def save_decoder(self, weight_file):
        AutoEncoderBuidler.__save_model(self.decoder, weight_file)

    @classmethod
    def __save_model(cls, model, weight_file):
        load_path = os.path.join(weight_file)
        print("save " + load_path)
        model.save(load_path)

    @classmethod
    def load_model(cls, weight_file):
        load_path = os.path.join(weight_file)
        print("load ", load_path)
        from keras.models import load_model
        return load_model(load_path)


def main():
    builded = AutoEncoderBuidler()

    builded.encoder.fit()
    builded.decoder.fit()
    builded.auto_encoder.fit()
    builded.auto_encoder.predict()


if __name__ == "__main__":
    main()
