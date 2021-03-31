from src.agents.models.generic import ResNetBlock
import keras
from keras.layers import Input, Flatten, Dense, ReLU, Convolution2D, MaxPool2D, Concatenate, BatchNormalization, \
    GlobalAvgPool2D
from keras.initializers import GlorotNormal


class ClassicConv(keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def build(self, input_shape):
        input_shape_2d, input_shape_1d = input_shape

        input_2d = Input(shape=input_shape_2d, name="input_2d")  # like 2d image plus channels
        input_1d = Input(shape=input_shape_1d, name="input_1d")  # like 1d vector

        conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(input_2d)
        norm1 = ReLU()(conv1)
        pool1 = MaxPool2D((2, 2))(norm1)

        conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(pool1)
        norm2 = ReLU()(conv2)
        pool2 = MaxPool2D((2, 2))(norm2)

        latent_conv = Flatten(name='latent')(pool2)
        concat = Concatenate()([latent_conv, input_1d])
        latent = Dense(self.latent_dim, activation="relu")(concat)

        self.inputs = [input_2d, input_1d]
        self.outputs = [latent]

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class ResNet(keras.Model):
    def __init__(self, latent_dim, latent_depth):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_depth = latent_depth

    def build(self, input_shape):
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

        self.inputs = [input_2d, input_1d]
        self.outputs = [latent]

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        return {"latent_dim": self.latent_dim,
                "latent_depth": self.latent_depth}
