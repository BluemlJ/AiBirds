from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, LeakyReLU, Convolution2D, BatchNormalization, \
    LayerNormalization, Concatenate, MaxPool2D
from tensorflow.keras.initializers import VarianceScaling, GlorotNormal

from src.envs import *


def get_input_model(env, latent_dim, latent_depth):
    if isinstance(env, AngryBirds):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_angry_birds_model(image_state_shape, latent_dim)
    elif isinstance(env, Tetris):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_tetris_model(image_state_shape, latent_dim)
    elif isinstance(env, Snake):
        image_state_shape, numerical_state_shape = env.get_state_shapes()
        return get_snake_model(image_state_shape, numerical_state_shape, latent_dim)
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


def get_snake_model(image_state_shape, numerical_state_shape, latent_dim):
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

    return [image_input, numeric_input], latent


def get_cb_model(image_state_shape, numerical_state_shape, latent_dim, latent_depth):
    image_input = Input(shape=image_state_shape, name="image_input")  # like 2d image plus channels
    numeric_input = Input(shape=numerical_state_shape, name="numeric_input")  # like 1d vector

    enc = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc1")(image_input)
    enc = Convolution2D(64, (2, 2), strides=2, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc2")(enc)
    enc = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc3")(enc)
    enc = Convolution2D(256, (2, 2), strides=2, padding='valid', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc4")(enc)

    latent_conv = Flatten(name='latent')(enc)
    latent = Concatenate()([latent_conv, numeric_input])
    for i in range(latent_depth):
        latent = Dense(latent_dim, activation="relu")(latent)

    return [image_input, numeric_input], latent
