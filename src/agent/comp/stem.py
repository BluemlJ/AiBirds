from abc import ABCMeta


class StemNetwork(metaclass=ABCMeta):
    def __init__(self, sequential, sequence_len=None):
        super(StemNetwork, self).__init__()
        self.sequential = sequential
        self.sequence_len = sequence_len
        self.hidden = None

    def get_functional_graph(self, input_shape, batch_size=None):
        pass

    def get_config(self):
        pass

    def get_hidden_state_shape(self):
        if self.sequential:
            return self.hidden.states[1].shape[-1]
        else:
            return None

    '''def get_cell_states(self):
        """Returns the model's LSTM cell states (if they exist)."""
        return None

    def set_cell_states(self, states):
        pass

    def reset_cell_states_for(self, instance_ids):
        pass'''
