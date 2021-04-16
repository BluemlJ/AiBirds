import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, LeakyReLU, Convolution2D, BatchNormalization, \
    LayerNormalization, Concatenate, MaxPool2D, Conv2DTranspose
from tensorflow.keras.initializers import GlorotNormal

# Suppress unnecessary TF warnings
tf.get_logger().setLevel('ERROR')


class ImageSequence(Sequence):
    def __init__(self, env, num_instances, batch_size):
        self.env = env
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.input_shape = env.get_state_shapes()
        self.data_2d, self.data_numeric = self.env.get_states()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_instances(self, ids):
        return self.data_2d[ids], self.data_numeric[ids]

    def __len__(self):
        return self.num_instances // self.batch_size

    def __getitem__(self, idx):
        range_min = idx * self.batch_size
        range_max = np.min([(idx + 1) * self.batch_size, self.num_instances])
        images = self.data_2d[range_min:range_max]
        numerics = self.data_numeric[range_min:range_max]
        return (images, numerics), (images, numerics)


class Autoencoder:
    def __init__(self, env):
        self.env = env
        self.model = None

    def pretrain_model(self, train_size, validation_size, batch_size=2048, epochs=5):
        """Pre-trains an autoencoder on randomly generated data given by the environment."""

        # Gather all data to use for training and validation
        print("Gathering data...")
        train_env = self.env(train_size)
        val_env = self.env(validation_size)
        training_batch_generator = ImageSequence(train_env, train_size, batch_size)
        validation_batch_generator = ImageSequence(val_env, validation_size, batch_size)

        # Initialize autoencoder (AE)
        print("Initializing model...")
        self.build_compile_model(training_batch_generator.input_shape)
        self.model.summary()

        # Train the CNN AE
        print("Starting pre-training...")
        history = self.model.fit(x=training_batch_generator,
                                 validation_data=validation_batch_generator,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle=True)
        print("Pre-training finished!")

        # Predict and print some examples
        self.predict_and_print_result(training_batch_generator, [0, 1], train_env)
        self.predict_and_print_result(validation_batch_generator, [0, 1], val_env)

        # Plot learning history
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
        plt.show()

        # Save the encoder part of the CNN AE
        save_path = "out/" + self.env.NAME + "/pretrained"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_weights(save_path + "/pretrained", overwrite=True, save_format="h5")

    def predict_and_print_result(self, generator, ids, env):
        test_data = generator.get_instances(ids)
        for test_2d, test_num in zip(*test_data):
            print("Input:\n" + env.state_2d_to_text(test_2d))
            print(env.state_1d_to_text(test_num))
            out_2d, out_num = self.model.predict([np.expand_dims(test_2d, axis=0), np.expand_dims(test_num, axis=0)])
            print("Output:\n" + env.state_2d_to_text(np.round(out_2d[0])))
            print(env.state_1d_to_text(np.round(out_num[0])) + "\n")

    def build_compile_model(self, input_shape):
        enc_num_dim = 8
        use_bias = False

        image_input = Input(shape=input_shape[0], name="image_input")
        numeric_input = Input(shape=input_shape[1], name="numeric_input")

        # ENCODING
        enc_2d = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                               use_bias=use_bias, activation="relu", name="enc_2d_1")(image_input)
        enc_2d = Convolution2D(32, (2, 2), strides=2, padding='same', kernel_initializer=GlorotNormal,
                               use_bias=use_bias, activation="relu", name="enc_2d_2")(enc_2d)
        enc_2d = Convolution2D(64, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                               use_bias=use_bias, activation="relu", name="enc_2d_3")(enc_2d)
        enc_2d = Convolution2D(64, (2, 2), strides=2, padding='valid', kernel_initializer=GlorotNormal,
                               use_bias=use_bias, activation="relu", name="enc_2d_4")(enc_2d)
        enc_2d_flat = Convolution2D(512, enc_2d.shape[1:3], strides=1, padding='valid', kernel_initializer=GlorotNormal,
                                    use_bias=use_bias, activation="relu", name="enc_2d_5")(enc_2d)
        enc_2d_flat = Flatten(name='enc_2d_flat')(enc_2d_flat)

        enc_num = Dense(enc_num_dim, activation="relu", name="enc_num")(numeric_input)

        enc = Concatenate(name="final_enc")([enc_2d_flat, enc_num])

        # DECODING
        dec_2d_flat, dec_num = tf.split(enc, (enc.shape[1] - enc_num_dim, enc_num_dim), axis=1)

        dec_num = Dense(input_shape[1], activation="relu", name="dec_num")(dec_num)

        dec_2d = tf.reshape(dec_2d_flat, shape=(-1, 1, 1, dec_2d_flat.shape[1]))
        dec_2d = Conv2DTranspose(64, enc_2d.shape[1:3], strides=1, padding='valid', kernel_initializer=GlorotNormal,
                                 use_bias=use_bias, activation="relu")(dec_2d)
        dec_2d = Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer=GlorotNormal,
                                 padding="valid", output_padding=(1, 0), activation='relu', use_bias=use_bias)(dec_2d)
        dec_2d = Conv2DTranspose(32, (2, 2), strides=1, kernel_initializer=GlorotNormal,
                                 padding="same", activation='relu', use_bias=use_bias)(dec_2d)
        dec_2d = Conv2DTranspose(32, (2, 2), strides=2, kernel_initializer=GlorotNormal,
                                 padding="same", activation='relu', use_bias=use_bias)(dec_2d)
        dec_2d = Conv2DTranspose(8, (4, 4), strides=1, kernel_initializer=GlorotNormal,
                                 padding="same", activation='relu', use_bias=use_bias, name="dec_2d")(dec_2d)

        # Compilation
        self.model = tf.keras.Model(inputs=[image_input, numeric_input], outputs=[dec_2d, dec_num])
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
