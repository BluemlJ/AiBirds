from abc import ABCMeta
from tensorflow.keras import Model


class StemModel(Model, metaclass=ABCMeta):
    def __init__(self, sequential, sequence_len=None):
        super(StemModel, self).__init__()
        self.sequential = sequential
        self.stateful = None
        self.sequence_len = sequence_len
        self.hidden = None

    def get_hidden_state_shape(self):
        if self.hidden is None:
            return None
        else:
            return self.hidden.states[1].shape[-1]

    def get_cell_states(self):
        """Returns the model's LSTM cell states (if they exist)."""
        return None

    def set_cell_states(self, states):
        pass

    def reset_cell_states_for(self, instance_ids):
        pass
