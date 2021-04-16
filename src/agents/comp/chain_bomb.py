from src.agents.comp.generic import ResNetBlock
from src.agents.comp.stem import StemNetwork
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, Convolution2D, Concatenate, BatchNormalization, \
    GlobalAvgPool2D
from tensorflow.keras.initializers import GlorotNormal


class ResNet(StemNetwork):
    def __init__(self, latent_dim, latent_depth):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim
        self.latent_depth = latent_depth

    def build(self, input_shape):
        input_shape_2d, input_shape_1d = input_shape

        enc_num_dim = 16
        num_channels = 64

        input_2d = Input(shape=input_shape_2d, name="input_2d")
        input_1d = Input(shape=input_shape_1d, name="input_1d")

        conv = Convolution2D(num_channels, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                             use_bias=False, name="conv_in")(input_2d)
        norm = BatchNormalization(name="norm_in")(conv)
        relu = ReLU(name="relu_in")(norm)

        res = ResNetBlock([32, 64, 128, 256], 4, name="block1", first_block=True)(relu)
        # res = ResNetBlock(256, 2, name="block2")(res)
        # res = ResNetBlock(256, 2, name="block3")(res)

        avg = GlobalAvgPool2D()(res)

        flat = Flatten(name='latent')(avg)

        num = Dense(enc_num_dim, activation="relu", name="enc_num")(input_1d)

        flat = Concatenate(name="final_enc")([flat, num])
        dense = Dense(self.latent_dim, activation="relu")(flat)
        for i in range(self.latent_depth - 1):
            dense = Dense(self.latent_dim, activation="relu")(dense)
        latent = dense

        self.inputs = [input_2d, input_1d]
        self.outputs = [latent]

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        return {"latent_dim": self.latent_dim,
                "latent_depth": self.latent_depth}
