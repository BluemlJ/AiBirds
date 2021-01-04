import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, LeakyReLU, Convolution2D, BatchNormalization, \
    LayerNormalization, Concatenate, MaxPool2D, Conv2DTranspose
from tensorflow.keras.initializers import GlorotNormal


class ImageSequence(Sequence):
    def __init__(self, env, num_instances, batch_size):
        self.env = env
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.input_shape, _ = env.get_state_shapes()
        self.data = self.env.get_states()[0]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return self.num_instances // self.batch_size

    def __getitem__(self, idx):
        range_min = idx * self.batch_size
        range_max = np.min([(idx + 1) * self.batch_size, self.num_instances])
        images = self.data[range_min:range_max]
        return images, images


def pretrain_model(env, train_size, validation_size):
    """Pre-trains an autoencoder on randomly generated data given by the environment."""

    batch_size = 128

    # Gather all data to use for training and validation
    print("Gathering data...")
    train_env = env(train_size)
    val_env = env(validation_size)
    training_batch_generator = ImageSequence(train_env, train_size, batch_size)
    validation_batch_generator = ImageSequence(val_env, validation_size, batch_size)

    # Initialize autoencoder (AE)
    print("Initializing model...")
    ae = build_compile_model(training_batch_generator.input_shape)
    ae.summary()

    # Train the CNN AE
    print("Starting pre-training...")
    history = ae.fit(x=training_batch_generator,
                     validation_data=validation_batch_generator,
                     epochs=5,
                     verbose=1,
                     shuffle=True)
    print("Pre-training finished!")

    # Predict and plot an example
    test = training_batch_generator[0][0][0]
    print("First train image:\n", train_env.image_state_to_text(test))
    out = ae.predict(np.expand_dims(test, axis=0))
    print("Output:\n", train_env.image_state_to_text(np.round(out[0])))

    test = validation_batch_generator[0][0][0]
    print("First validation image:\n", val_env.image_state_to_text(test))
    out = ae.predict(np.expand_dims(test, axis=0))
    print("Output:\n", val_env.image_state_to_text(np.round(out[0])))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    # Save the encoder part of the CNN AE
    save_path = "out/" + env.NAME + "/pretrained"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ae.save_weights(save_path + "/pretrained", overwrite=True, save_format="h5")


def build_compile_model(input_shape):
    image_input = Input(shape=input_shape, name="image_input")  # like 2d image plus channels

    # ENCODING
    enc = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc1")(image_input)
    enc = Convolution2D(64, (2, 2), strides=2, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc2")(enc)
    enc = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc3")(enc)
    enc = Convolution2D(256, (2, 2), strides=2, padding='valid', kernel_initializer=GlorotNormal,
                        use_bias=False, activation="relu", name="enc4")(enc)

    # DECODING
    dec = Conv2DTranspose(128, (2, 2), strides=2, kernel_initializer=GlorotNormal,
                          padding="valid", output_padding=(1, 0), activation='relu', use_bias=False, name="dec1")(enc)
    dec = Conv2DTranspose(64, (2, 2), strides=1, kernel_initializer=GlorotNormal,
                          padding="same", activation='relu', use_bias=False, name="dec2")(dec)
    dec = Conv2DTranspose(32, (2, 2), strides=2, kernel_initializer=GlorotNormal,
                          padding="same", activation='relu', use_bias=False, name="dec3")(dec)
    dec = Conv2DTranspose(8, (4, 4), strides=1, kernel_initializer=GlorotNormal,
                          padding="same", activation='relu', use_bias=False, name="dec4")(dec)

    # Compilation
    model = tf.keras.Model(inputs=[image_input], outputs=[dec])
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model
