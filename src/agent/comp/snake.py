from src.agent.comp.generic import ResNetBlock
from src.agent.comp.stem import StemNetwork
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, Convolution2D, Concatenate, BatchNormalization, \
    GlobalAvgPool2D
from tensorflow.keras.initializers import GlorotNormal


class MyNet(StemNetwork):
    def __init__(self, latent_dim):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim

        self.conv_global = None
        self.conv_regional_1 = None
        self.conv_regional_2 = None
        self.conv_local = None

        self.dense = Dense(self.latent_dim, activation="relu")

    def get_functional_graph(self, input_shape, batch_size=None):
        input_shape_2d, input_shape_1d = input_shape

        input_2d = Input(shape=input_shape_2d, name="input_2d")
        input_1d = Input(shape=input_shape_1d, name="input_1d")

        conv_global = Convolution2D(32, input_shape_2d[-3:-1], use_bias=False,
                                    kernel_initializer=GlorotNormal, name="conv_global")(input_2d)
        glob = GlobalAvgPool2D()(conv_global)
        conv_regional_1 = Convolution2D(32, (5, 5), strides=1, use_bias=False, padding='same',
                                        kernel_initializer=GlorotNormal, name="conv_regional_1")(input_2d)
        conv_regional_2 = Convolution2D(128, (2, 2), strides=1, use_bias=False, padding='same',
                                        kernel_initializer=GlorotNormal, name="conv_regional_2")(conv_regional_1)
        reg = GlobalAvgPool2D()(conv_regional_2)
        conv_local = Convolution2D(64, (3, 3), strides=1, use_bias=False,
                                   kernel_initializer=GlorotNormal, name="conv_local")(input_2d)
        loc = GlobalAvgPool2D()(conv_local)

        concat = Concatenate()([glob, reg, loc, input_1d])
        latent = Dense(self.latent_dim, activation="relu")(concat)

        return [input_2d, input_1d], latent

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class ResNet(StemNetwork):
    def __init__(self, latent_dim, latent_depth):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim
        self.latent_depth = latent_depth

    def get_functional_graph(self, input_shape, batch_size=None):
        input_shape_2d, input_shape_1d = input_shape

        enc_num_dim = 16
        num_channels = 32

        input_2d = Input(shape=input_shape_2d, name="input_2d")
        input_1d = Input(shape=input_shape_1d, name="input_1d")

        conv = Convolution2D(num_channels, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                             use_bias=False, name="conv_in")(input_2d)
        norm = BatchNormalization(name="norm_in")(conv)
        relu = ReLU(name="relu_in")(norm)

        res = ResNetBlock(32, 1, name="block1", first_block=True)(relu)
        res = ResNetBlock(64, 1, name="block2")(res)
        res = ResNetBlock(128, 1, name="block3")(res)

        avg = GlobalAvgPool2D(name="ResNet_out")(res)

        flat = Flatten(name='latent')(avg)

        num = Dense(enc_num_dim, activation="relu", name="enc_num")(input_1d)

        flat = Concatenate(name="final_enc")([flat, num])
        dense = Dense(self.latent_dim, activation="relu")(flat)
        for i in range(self.latent_depth - 1):
            dense = Dense(self.latent_dim, activation="relu")(dense)
        latent = dense

        return [input_2d, input_1d], latent

    def get_config(self):
        return {"latent_dim": self.latent_dim,
                "latent_depth": self.latent_depth}
