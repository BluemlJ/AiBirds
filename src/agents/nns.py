import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, LeakyReLU, Convolution2D, BatchNormalization, \
    LayerNormalization, Concatenate, MaxPool2D, Add, GlobalAvgPool2D, Layer
from tensorflow.keras.initializers import VarianceScaling, GlorotNormal
from tensorflow.keras.regularizers import l2

from src.envs import *


class Residual(tf.keras.Model):
    """The Residual block of ResNet."""

    def __init__(self, num_channels, name, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Convolution2D(num_channels, padding='same', kernel_size=3,
                                   strides=strides, name=name + "_conv1")
        self.conv2 = Convolution2D(num_channels, kernel_size=3, padding='same',
                                   name=name + "_conv1")
        self.skip_conv = None
        if use_1x1conv:
            self.skip_conv = Convolution2D(num_channels, kernel_size=1, strides=strides,
                                           name=name + "_skip_conv")
        else:
            self.skip_conv = Layer()
        self.bn1 = BatchNormalization(name=name + "_bn1")
        self.bn2 = BatchNormalization(name=name + "_bn2")

    def call(self, inputs, **kwargs):
        x = ReLU()(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        inputs = self.skip_conv(inputs)
        x += inputs
        return ReLU()(x)

    def get_config(self):
        pass


class ResNetBlock(Layer):
    def __init__(self, num_channels, num_residuals, name, first_block=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.residuals = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residuals.append(Residual(num_channels, use_1x1conv=True, strides=2,
                                               name=name + "_res%d" % i))
            else:
                self.residuals.append(Residual(num_channels, name=name + "_res%d" % i))

    def call(self, x, **kwargs):
        for residual in self.residuals:
            x = residual(x)
        return x


def get_input_model(env, latent_dim, latent_depth):
    if isinstance(env, AngryBirds):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_angry_birds_model(image_state_shape, latent_dim)
    elif isinstance(env, Tetris):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_tetris_model(image_state_shape, latent_dim)
    elif isinstance(env, Snake):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_snake_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth)
    elif isinstance(env, ChainBomb):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_cb_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth)
    else:
        raise ValueError("ERROR: Invalid environment given.")


def get_angry_birds_model(state_shape, latent_dim):
    input_frame = Input(shape=state_shape)

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
    latent = Dense(latent_dim, activation="relu")(latent)

    return [input_frame], latent


def get_tetris_model(state_shape, latent_dim):
    input_frame = Input(shape=state_shape)

    conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(input_frame)
    norm1 = ReLU()(conv1)
    pool1 = MaxPool2D((2, 2))(norm1)

    conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(pool1)
    norm2 = ReLU()(conv2)
    pool2 = MaxPool2D((2, 2))(norm2)

    latent = Flatten(name='latent')(pool2)
    latent = Dense(latent_dim, activation="relu")(latent)

    return [input_frame], latent


def get_snake_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth):
    enc_num_dim = 16
    num_channels = 64

    image_input = Input(shape=image_state_shape, name="image_input")
    numeric_input = Input(shape=numerical_state_shape, name="numeric_input")

    conv = Convolution2D(num_channels, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                         use_bias=False, name="conv_in")(image_input)
    norm = BatchNormalization(name="norm_in")(conv)
    relu = ReLU(name="relu_in")(norm)

    res = ResNetBlock(64, 2, name="block1", first_block=True)(relu)
    res = ResNetBlock(256, 2, name="block2")(res)
    # res = ResNetBlock(256, 2, name="block3")(res)

    avg = GlobalAvgPool2D()(res)

    '''conv = Convolution2D(1, (1, 1), strides=1, padding='same', kernel_initializer=GlorotNormal,
                         use_bias=False, name="conv_mid")(res)
    norm = BatchNormalization(name="norm_mid")(conv)
    relu = ReLU(name="relu_mid")(norm)'''

    flat = Flatten(name='latent')(avg)

    num = Dense(enc_num_dim, activation="relu", name="enc_num")(numeric_input)

    flat = Concatenate(name="final_enc")([flat, num])
    dense = Dense(latent_dim, activation="relu")(flat)
    for i in range(latent_depth - 1):
        dense = Dense(latent_dim, activation="relu")(dense)

    return [image_input, numeric_input], dense


'''def get_snake_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth):
    image_input = Input(shape=image_state_shape, name="image_input")  # like 2d image plus channels
    numeric_input = Input(shape=numerical_state_shape, name="numeric_input")  # like 1d vector

    conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(image_input)
    norm1 = ReLU()(conv1)
    pool1 = MaxPool2D((2, 2))(norm1)

    conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(pool1)
    norm2 = ReLU()(conv2)
    pool2 = MaxPool2D((2, 2))(norm2)

    latent_conv = Flatten(name='latent')(pool2)
    concat = Concatenate()([latent_conv, numeric_input])
    latent = Dense(latent_dim, activation="relu")(concat)

    return [image_input, numeric_input], latent'''


def get_cb_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth):
    enc_num_dim = 16
    filters = 64

    image_input = Input(shape=image_state_shape, name="image_input")
    numeric_input = Input(shape=numerical_state_shape, name="numeric_input")

    conv = Convolution2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=GlorotNormal,
                         use_bias=False, name="conv_in")(image_input)
    norm = BatchNormalization(name="norm_in")(conv)
    relu = ReLU(name="relu_in")(norm)

    res_2d = get_2d_residual_block(relu, filters, name="2d_residual_block_1")
    res_2d = get_2d_residual_block(res_2d, filters, name="2d_residual_block_2")
    # res_2d = get_2d_residual_block(res_2d, filters, name="2d_residual_block_3")
    # res_2d = get_2d_residual_block(res_2d, filters, name="2d_residual_block_4")

    conv = Convolution2D(1, (1, 1), strides=1, padding='same', kernel_initializer=GlorotNormal,
                         use_bias=False, name="conv_mid")(res_2d)
    norm = BatchNormalization(name="norm_mid")(conv)
    relu = ReLU(name="relu_mid")(norm)

    flat_2d = Flatten(name='latent')(relu)

    num = Dense(enc_num_dim, activation="relu", name="enc_num")(numeric_input)

    flat = Concatenate(name="final_enc")([flat_2d, num])
    dense = Dense(latent_dim, activation="relu")(flat)
    for i in range(latent_depth - 1):
        dense = Dense(latent_dim, activation="relu")(dense)

    return [image_input, numeric_input], dense

    return [image_input, numeric_input], latent
