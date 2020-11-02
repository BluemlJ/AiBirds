from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, LeakyReLU, Convolution2D, BatchNormalization, \
    LayerNormalization, Concatenate
from tensorflow.keras.initializers import VarianceScaling, GlorotNormal
import numpy as np

from src.envs.angry_birds import AngryBirds
from src.envs.tetris import Tetris
from src.envs.snake import Snake


def get_input_model(env, latent_dim):
    if isinstance(env, AngryBirds):
        image_state_shape, numerical_state_shape = env.get_state_shape()
        return get_angry_birds_model(image_state_shape, latent_dim)
    elif isinstance(env, Tetris):
        image_state_shape, numerical_state_shape = env.get_state_shape()
        return get_tetris_model(image_state_shape, latent_dim)
    elif isinstance(env, Snake):
        image_state_shape, numerical_state_shape = env.get_state_shape()
        return get_snake_model(image_state_shape, numerical_state_shape, latent_dim)
    else:
        raise ValueError("ERROR: Invalid environment given.")


def get_angry_birds_model(state_shape, latent_dim):
    input_frame = Input(shape=state_shape)

    conv1 = Convolution2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False)(input_frame)
    conv1_norm = ReLU()(BatchNormalization()(conv1))

    conv2 = Convolution2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False)(conv1_norm)
    conv2_norm = ReLU()(BatchNormalization()(conv2))

    conv3 = Convolution2D(64, (2, 2), strides=2, kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False)(conv2_norm)
    conv3_norm = ReLU()(BatchNormalization()(conv3))

    conv4 = Convolution2D(latent_dim, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.),
                          activation='relu',
                          use_bias=False,
                          name='final_conv')(conv3_norm)
    conv4_norm = ReLU()(BatchNormalization()(conv4))

    latent = Flatten(name='latent')(conv4_norm)

    return [input_frame], latent


def get_tetris_model(state_shape, latent_dim):
    input_frame = Input(shape=state_shape)

    conv1 = Convolution2D(16, (3, 3), strides=1, padding='same', kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False)(input_frame)
    conv1_norm = ReLU()(conv1)

    conv2 = Convolution2D(32, (3, 3), strides=1, padding='same', kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False)(conv1_norm)
    conv2_norm = ReLU()(conv2)

    conv3 = Convolution2D(latent_dim, state_shape[:-1], kernel_initializer=VarianceScaling(scale=2.),
                          use_bias=False, name='final_conv')(conv2_norm)
    conv3_norm = ReLU()(conv3)

    latent = Flatten(name='latent')(conv3_norm)

    return [input_frame], latent


def get_snake_model(image_state_shape, numerical_state_shape, latent_dim):
    image_input = Input(shape=image_state_shape, name="image_input")  # like 2d image plus channels
    numeric_input = Input(shape=numerical_state_shape, name="numeric_input")  # like 1d vector

    conv1 = Convolution2D(16, (3, 3), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(image_input)
    norm1 = ReLU()(conv1)

    conv2 = Convolution2D(32, (3, 3), strides=1, padding='same', kernel_initializer=GlorotNormal,
                          use_bias=False)(norm1)
    norm2 = ReLU()(conv2)

    conv3 = Convolution2D(latent_dim, image_state_shape[:-1], kernel_initializer=GlorotNormal,
                          use_bias=False, name='final_conv')(norm2)
    norm3 = ReLU()(conv3)

    latent_conv = Flatten(name='latent')(norm3)
    concat = Concatenate()([latent_conv, numeric_input])
    latent = Dense(latent_dim, activation="relu")(concat)

    return [image_input, numeric_input], latent
