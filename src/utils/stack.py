import numpy as np
from src.utils.utils import shapes2arrays, increase_last_dim


class StateStacker:
    """Conveniently manages frames/states for frame-stacking."""

    def __init__(self, state_shapes, stack_size, num_par_inst):
        self.stack_size = stack_size
        self.state_shapes = state_shapes
        self.stack_shapes = increase_last_dim(state_shapes, factor=stack_size)
        self.stacks = shapes2arrays(self.stack_shapes, preceded_by=(num_par_inst,))
        self.stack_empty = np.ones(num_par_inst, dtype="bool")

    def get_stack_shapes(self):
        return self.stack_shapes

    def add_states(self, states):
        self.shift_back()
        for i in range(len(self.stack_shapes)):
            self.stacks[i][self.stack_empty] = np.tile(states[i][self.stack_empty], self.stack_size)
            frame_size = self.state_shapes[i][-1]
            self.stacks[i][~ self.stack_empty, ..., -frame_size:] = states[i][~ self.stack_empty]
        self.stack_empty[:] = False

    def shift_back(self):
        if self.stack_size == 1:
            return
        for i in range(len(self.stack_shapes)):
            frame_size = self.state_shapes[i][-1]
            self.stacks[i][..., :-frame_size] = self.stacks[i][..., frame_size:]

    def get_stacks(self):
        assert not np.any(self.stack_empty)
        return self.stacks

    def reset_stacks(self, ids):
        self.stack_empty[ids] = True

    def get_frame(self, frame_id, env_id=None):
        """Returns a frame from the given position ID.
        :param frame_id: int between 0 and stack_size (the larger, the older; 0 = current)
        :param env_id: int between 0 and num_par_envs
        """
        frame = []
        for i in range(len(self.stack_shapes)):
            frame_size = self.state_shapes[i][-1]
            frame_window = np.arange(frame_size) - (frame_id + 1) * frame_size
            if env_id is None:
                frame += [self.stacks[i][..., frame_window]]
            else:
                frame += [self.stacks[i][env_id][..., frame_window]]
        return frame
