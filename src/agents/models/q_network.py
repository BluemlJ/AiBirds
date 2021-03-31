from abc import ABCMeta

import keras
import tensorflow as tf
from keras.layers import Dense, LeakyReLU


class QNetwork(keras.Model, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(QNetwork, self).__init__(**kwargs)
        self.num_actions = None

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

    def check_initialization(self):
        if self.num_actions is None:
            raise ValueError("You must specify number of actions first before building the Q-network.")


class DoubleQNetwork(QNetwork):
    def __init__(self, latent_v_dim, latent_a_dim):
        super().__init__(name="double_Q_network")
        self.latent_v_dim = latent_v_dim
        self.latent_a_dim = latent_a_dim

    def build(self, input_shape=None):
        self.check_initialization()

        # State value
        self.latent_v = Dense(self.latent_v_dim, name='latent_V', activation="relu")
        self.state_value = Dense(1, name='V')

        # Advantage
        self.latent_a = Dense(self.latent_a_dim, name='latent_A', activation="relu")
        self.advantage = Dense(self.num_actions, name='A')

    def get_config(self):
        config = {"latent_v_dim": self.latent_v_dim,
                  "latent_a_dim": self.latent_a_dim}
        return config

    def call(self, inputs, training=None, mask=None):
        v = self.state_value(self.latent_v(inputs))
        a = self.advantage(self.latent_a(inputs))
        a_avg = tf.reduce_mean(a, axis=1, keepdims=True, name='A_mean')
        # State-action values: Q(s, a) = V(s) + A(s, a) - A_mean(s, a)
        return v + a - a_avg


class VanillaQNetwork(QNetwork):
    def __init__(self, num_actions):
        super().__init__(name="default_Q_network")
        self.num_actions = num_actions

    def build(self, input_shape=None):
        self.check_initialization()

        self.dense1 = Dense(128, name='latent_2')
        self.lrelu = LeakyReLU()
        self.dense2 = Dense(self.num_actions, name='Q')

    def get_config(self):
        return {}

    def call(self, inputs, training=None, mask=None):
        latent_q = self.lrelu(self.dense1(inputs))
        return self.dense2(latent_q)
