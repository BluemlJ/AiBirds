import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class MyNoisyDense(Layer):
    """Own creation of a noisy dense layer."""

    def __init__(self, size, std_init=0.5, **kwargs):
        super().__init__(**kwargs)
        self.size_in = None
        self.size_out = size

        self.std_init = std_init
        self.noise_active = True

        self.mu = None
        self.sigma = None

    def build(self, input_shape):
        self.size_in = input_shape[-1]
        layer_shape = (self.size_out, self.size_in)

        norm = 1 / self.size_in  # Why sqrt?

        init_mu = tf.random.uniform(layer_shape) * norm
        self.mu = tf.Variable(init_mu, dtype="float32", trainable=True)

        init_sigma = tf.random.uniform(layer_shape) * self.std_init * norm
        self.sigma = tf.Variable(init_sigma, dtype="float32", trainable=True)

        self.reset_noise()

    def reset_noise(self):
        pass

    def set_noisy(self, active: bool):
        self.noise_active = active

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self.noise_active:
            noise = tf.random.normal((self.size_out,))
            return tf.linalg.matvec(self.mu, x) + noise
        else:
            return tf.linalg.matvec(self.mu, x)

    def get_config(self):
        return {"size": self.size_out,
                "std_init": self.std_init}


def _get_noise_vector(size):
    x = tf.random.normal((size,))
    return tf.sign(x) * tf.sqrt(tf.abs(x))


class NoisyDense(Layer):
    """Part of Noisy Nets, as used in Rainbow paper."""

    def __init__(self, size, std_init=0.5, **kwargs):
        super().__init__(**kwargs)
        self.size_in = None
        self.size_out = size
        self.std_init = std_init
        self.noise_active = True

        self.mu = None
        self.mu_bias = None
        self.sigma = None
        self.sigma_bias = None
        self.epsilon = None
        self.epsilon_bias = None

    def build(self, input_shape):
        self.size_in = input_shape[-1]

        norm = 1 / np.sqrt(self.size_in)
        uniform_init = tf.random_uniform_initializer(-norm, norm)

        self.mu = tf.Variable(uniform_init(shape=(self.size_out, self.size_in), dtype="float32"),
                              trainable=True)
        self.mu_bias = tf.Variable(initial_value=tf.ones(self.size_out) * self.std_init * norm,
                                   dtype="float32", trainable=True)
        self.sigma = tf.Variable(uniform_init(shape=(self.size_out, self.size_in), dtype="float32"),
                                 trainable=True)
        self.sigma_bias = tf.Variable(initial_value=tf.ones(self.size_out) * self.std_init * norm,
                                      dtype="float32", trainable=True)
        self.epsilon = tf.Variable(initial_value=tf.zeros((self.size_out, self.size_in)), dtype="float32",
                                   trainable=False)
        self.epsilon_bias = tf.Variable(initial_value=tf.zeros(self.size_out), dtype="float32", trainable=False)
        self.reset_noise()

    def reset_noise(self):
        epsilon_in = _get_noise_vector(self.size_in)
        epsilon_out = _get_noise_vector(self.size_out)
        self.epsilon.assign(tf.tensordot(epsilon_out, epsilon_in, axes=0))  # outer product
        self.epsilon_bias.assign(epsilon_out)

    def set_noisy(self, active: bool):
        # self.epsilon.assign(tf.zeros(self.epsilon.get_shape()))
        # self.epsilon_bias.assign(tf.zeros(self.epsilon_bias.get_shape()))
        self.noise_active = active

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self.noise_active:
            A = self.mu + self.sigma * self.epsilon
            b = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            A = self.mu
            b = self.mu_bias
        return tf.linalg.matvec(A, x) + b

    def get_config(self):
        return {"size": self.size_out,
                "std_init": self.std_init}
