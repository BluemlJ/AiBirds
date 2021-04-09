from tensorflow.keras.layers import ReLU, Convolution2D, BatchNormalization, Layer, Flatten, Dense,\
    MaxPool2D, Concatenate, LSTM, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.initializers import GlorotNormal
import tensorflow as tf
from src.agents.comp.stem import StemModel


class ConvHead(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                                   use_bias=False, activation="relu")
        self.conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                                   use_bias=False, activation="relu")
        self.flat = Flatten(name='latent')

    def call(self, inputs, training=None, mask=None):
        conv1 = MaxPool2D((2, 2))(self.conv1(inputs))
        conv2 = MaxPool2D((2, 2))(self.conv2(conv1))
        return self.flat(conv2)

    def get_config(self):
        return {}


class TimeConvHead(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Convolution2D(32, (4, 4), strides=1, padding='same', kernel_initializer=GlorotNormal,
                                   use_bias=False, activation="relu")
        self.conv2 = Convolution2D(128, (2, 2), strides=1, padding='same', kernel_initializer=GlorotNormal,
                                   use_bias=False, activation="relu")
        self.flat = Flatten(name='latent')

    def call(self, inputs, training=None, mask=None):
        conv1 = TimeDistributed(MaxPool2D((2, 2)))(TimeDistributed(self.conv1)(inputs))
        conv2 = TimeDistributed(MaxPool2D((2, 2)))(TimeDistributed(self.conv2)(conv1))
        return TimeDistributed(self.flat)(conv2)

    def get_config(self):
        return {}


class ConvStem(StemModel):
    def __init__(self, latent_dim):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim
        self.conv_head = ConvHead()
        self.latent = Dense(self.latent_dim, activation="relu")

    def call(self, inputs, training=None, mask=None):
        input_2d, input_1d = inputs
        conv = self.conv_head(input_2d)
        concat = Concatenate()([conv, input_1d])
        return self.latent(concat)

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class ConvLSTM(StemModel):
    def __init__(self, latent_dim, lstm_dim, sequence_len):
        super().__init__(sequential=True, sequence_len=sequence_len)
        assert sequence_len >= 2
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim

        self.conv_head = TimeConvHead()
        self.lstm = LSTM(self.lstm_dim, stateful=True, return_sequences=True, name="lstm")
        self.latent = Dense(self.latent_dim, activation="relu")
        self.hidden = self.lstm

    def call(self, inputs, training=None, mask=None):
        mask = self.compute_mask(inputs)
        input_2d, input_1d = inputs
        conv = self.conv_head(input_2d)
        concat = Concatenate()([conv, input_1d])
        latent = self.latent(concat)
        lstm = self.lstm(latent, mask=mask)
        return lstm

    def compute_mask(self, inputs, mask=None):
        input_2d, input_1d = inputs
        mask_2d = tf.math.equal(input_2d, 0)
        mask_1d = tf.math.equal(input_1d, 0)
        mask_2d = tf.reduce_any(mask_2d, axis=(2, 3, 4))
        mask_1d = tf.reduce_any(mask_1d, axis=2)
        return tf.math.logical_and(mask_2d, mask_1d)

    def get_config(self):
        return {"latent_dim": self.latent_dim,
                "lstm_dim": self.lstm_dim,
                "sequence_len": self.sequence_len}

    def get_cell_states(self):
        return self.lstm.states[1].numpy()

    def set_cell_states(self, states):
        self.lstm.states[1].assign(states)

    def reset_cell_states_for(self, instance_ids):
        hidden_states = self.lstm.states[0].numpy()
        cell_states = self.lstm.states[1].numpy()
        hidden_states[instance_ids] = 0
        cell_states[instance_ids] = 0
        self.lstm.states[0].assign(hidden_states)
        self.lstm.states[1].assign(cell_states)


class Residual(Layer):
    """The elementary part of a Residual Network."""

    def __init__(self, num_channels, name, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Convolution2D(num_channels, kernel_size=3, padding='same', strides=strides,
                                   name=name + "_conv1", kernel_initializer=GlorotNormal)
        self.conv2 = Convolution2D(num_channels, kernel_size=3, padding='same',
                                   name=name + "_conv2", kernel_initializer=GlorotNormal)
        self.bn1 = BatchNormalization(name=name + "_bn1")
        self.bn2 = BatchNormalization(name=name + "_bn2")
        self.skip_conv = None
        if use_1x1conv:
            self.skip_conv = Convolution2D(num_channels, kernel_size=1, strides=strides,
                                           name=name + "_skip_conv", kernel_initializer=GlorotNormal)
        else:
            self.skip_conv = Layer()

    def call(self, inputs, **kwargs):
        x = ReLU()(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        skip = self.skip_conv(inputs)
        x += skip
        return ReLU()(x)


class ResNetBlock(Layer):
    """Sequence of residuals."""

    def __init__(self, num_channels, num_residuals, name, first_block=False, **kwargs):
        """Constructor

        :param num_channels: int
        :param num_residuals: number of residuals in this block
        :param name:
        :param first_block:
        :param kwargs:
        """

        super().__init__(name=name, **kwargs)

        self.residuals = []

        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residuals.append(Residual(num_channels, use_1x1conv=True,  # strides=2,
                                               name=name + "_res%d" % i))
            else:
                self.residuals.append(Residual(num_channels, name=name + "_res%d" % i))

    def call(self, x, **kwargs):
        for residual in self.residuals:
            x = residual(x)
        return x
