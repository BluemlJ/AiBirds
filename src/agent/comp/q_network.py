import tensorflow as tf
from abc import ABCMeta
from tensorflow.keras.layers import Dense, LeakyReLU, Layer, ReLU
from src.agent.comp.noisy import NoisyDense


class QNetwork(Layer, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(QNetwork, self).__init__(**kwargs)
        self.num_actions = None
        self.sequential = None
        self.q_value_axis = None

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

    def set_sequential(self, sequential):
        self.sequential = sequential
        if sequential:
            self.q_value_axis = 2
        else:
            self.q_value_axis = 1

    def check_initialization(self):
        if self.num_actions is None:
            raise ValueError("You must specify number of actions first before building the Q-network.")
        elif self.sequential is None:
            raise ValueError("You must set (non-)sequential before building the Q-network.")


class DoubleQNetwork(QNetwork):
    def __init__(self, latent_v_dim=None, latent_a_dim=None, noise_std_init=0):
        super().__init__(name="double_Q_network")
        self.v_h_size = latent_v_dim
        self.a_h_size = latent_a_dim
        self.noise_std_init = noise_std_init
        self.v_h = None
        self.v = None
        self.a_h = None
        self.a = None

    def build(self, input_shape=None):
        self.check_initialization()

        # State value V
        if self.v_h_size is not None:
            if self.noise_std_init == 0:
                self.v_h = Dense(self.v_h_size, name='latent_V')
            else:
                self.v_h = NoisyDense(self.v_h_size, std_init=self.noise_std_init, name='latent_V')
        else:
            self.v_h = Layer()

        if self.noise_std_init == 0:
            self.v = Dense(1, name='V')
        else:
            self.v = NoisyDense(1, self.noise_std_init, name='V')

        # Advantage A
        if self.a_h_size is not None:
            if self.noise_std_init == 0:
                self.a_h = Dense(self.a_h_size, name='latent_A')
            else:
                self.a_h = NoisyDense(self.a_h_size, std_init=self.noise_std_init, name='latent_A')
        else:
            self.a_h = Layer()

        if self.noise_std_init == 0:
            self.a = Dense(self.num_actions, name='A')
        else:
            self.a = NoisyDense(self.num_actions, self.noise_std_init, name='A')
        
        super(DoubleQNetwork, self).build(input_shape)

    def get_config(self):
        config = {"latent_v_dim": self.v_h_size,
                  "latent_a_dim": self.a_h_size,
                  "noise_std_init": self.noise_std_init}
        return config

    def call(self, inputs, training=False, mask=None):
        v = self.v(ReLU()(self.v_h(inputs)))
        a = self.a(ReLU()(self.a_h(inputs)))
        a_avg = tf.reduce_mean(a, axis=self.q_value_axis, keepdims=True, name='A_mean')
        # State-action values: Q(s, a) = V(s) + A(s, a) - A_mean(s, a)
        return v + a - a_avg

    def reset_noise(self):
        if self.noise_std_init > 0:
            self.v_h.reset_noise()
            self.v.reset_noise()
            self.a_h.reset_noise()
            self.a.reset_noise()

    def set_noisy(self, active):
        if self.noise_std_init > 0:
            self.v_h.set_noisy(active)
            self.v.set_noisy(active)
            self.a_h.set_noisy(active)
            self.a.set_noisy(active)


class VanillaQNetwork(QNetwork):
    def __init__(self):
        super().__init__(name="default_Q_network")

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
