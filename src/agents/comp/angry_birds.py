from src.agents.comp.stem import StemNetwork
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, Convolution2D, MaxPool2D
from tensorflow.keras.initializers import GlorotNormal


class ClassicConv(StemNetwork):
    def __init__(self, latent_dim):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        input_frame = Input(shape=input_shape)

        conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(input_frame)
        norm1 = ReLU()(conv1)
        pool1 = MaxPool2D((2, 2))(norm1)

        conv2 = Convolution2D(64, (3, 3), strides=2, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(pool1)
        norm2 = ReLU()(conv2)
        pool2 = MaxPool2D((2, 2))(norm2)

        conv3 = Convolution2D(64, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(pool2)
        norm3 = ReLU()(conv3)
        pool3 = MaxPool2D((2, 2))(norm3)

        conv4 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                              use_bias=False)(pool3)
        norm4 = ReLU()(conv4)
        pool4 = MaxPool2D((2, 2))(norm4)

        latent = Flatten(name='latent')(pool4)
        latent = Dense(self.latent_dim, activation="relu")(latent)

        self.inputs = [input_frame]
        self.outputs = [latent]

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass
