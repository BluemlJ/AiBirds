import keras
from keras.layers import Input, Flatten, Dense, ReLU, Convolution2D, MaxPool2D
from keras.initializers import GlorotNormal


class ClassicConv(keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def build(self, input_shape):
        input_shape_2d, _ = input_shape

        input_frame = Input(shape=input_shape_2d)

        conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(input_frame)
        norm1 = ReLU()(conv1)
        pool1 = MaxPool2D((2, 2))(norm1)

        conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(pool1)
        norm2 = ReLU()(conv2)
        pool2 = MaxPool2D((2, 2))(norm2)

        latent = Flatten(name='latent')(pool2)
        latent = Dense(self.latent_dim, activation="relu")(latent)

        self.inputs = [input_frame]
        self.outputs = [latent]

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        return {"latent_dim": self.latent_dim}
