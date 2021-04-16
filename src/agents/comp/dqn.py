import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.keras.utils import layer_utils
from src.agents.comp.stem import StemNetwork
from src.agents.comp.q_network import QNetwork


class DQN(Model):
    def __init__(self, stem_model: StemNetwork, q_network: QNetwork,
                 input_shape, num_actions, batch_size=None):
        super(DQN, self).__init__()
        self.stem_model = stem_model
        self.q_network = q_network
        self.sequential = stem_model.sequential
        self.num_actions = num_actions
        self.batch_size = batch_size if self.sequential else None
        self.build(input_shape)

    def build(self, input_shape):
        if self.stem_model.sequential:
            assert self.batch_size is not None

        self.q_network.set_num_actions(self.num_actions)
        self.q_network.set_sequential(self.sequential)
        self.q_network.build()

        input_shape_2d, input_shape_1d = input_shape
        input_shape_2d = (self.batch_size,) + input_shape_2d
        input_shape_1d = (self.batch_size,) + (input_shape_1d,)
        
        super(DQN, self).build([input_shape_2d, input_shape_1d])
        '''self.stem_model.build([input_shape_2d, input_shape_1d])'''

        # input_shape_2d, input_shape_1d = input_shape
        # if self.stem_model.sequential:  # variable-length sequences with fixed batch size
        #     assert self.batch_size is not None
        #     input_shape_2d = (None,) + input_shape_2d
        #     input_shape_1d = (None,) + (input_shape_1d,)
        # self.inputs = [Input(shape=input_shape_2d, batch_size=self.batch_size, name="input_2d"),
        #                Input(shape=input_shape_1d, batch_size=self.batch_size, name="input_1d")]

        # self.outputs = self.call(self.inputs)
        # self.built = True

    def call(self, inputs, training=None, mask=None):
        input_2d, input_1d = inputs
        input_2d = tf.cast(input_2d, dtype="float32")
        input_1d = tf.cast(input_1d, dtype="float32")
        latent = self.stem_model([input_2d, input_1d], mask=mask)
        q_vals = self.q_network(latent)
        return q_vals

    def get_config(self):
        pass

    '''def summary(self, line_length=None, positions=None, print_fn=None):
        layer_utils.print_summary(self,
                                  line_length=line_length,
                                  positions=positions,
                                  print_fn=print_fn)
        self.stem_model.summary(line_length=line_length,
                                positions=positions,
                                print_fn=print_fn)
        self.q_network.summary(line_length=line_length,
                               positions=positions,
                               print_fn=print_fn)'''

    def set_hidden_and_predict(self, hidden_states, inputs):
        assert self.stem_model.sequential
        input_2d, input_1d = inputs
        assert len(input_2d) % self.batch_size == 0

        predictions = np.empty(shape=(0, input_2d.shape[1], self.q_network.num_actions))
        data = (input_2d, input_1d, hidden_states)
        data = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)

        for step, (input_2d_b, input_1d_b, hidden_states_b) in enumerate(data):
            self.reset_states()
            self.stem_model.set_cell_states(hidden_states_b)
            batch_out = self.predict([input_2d_b, input_1d_b], batch_size=len(input_2d))  # TODO: de heck? Why bs?
            predictions = np.append(predictions, batch_out, axis=0)

        return predictions
