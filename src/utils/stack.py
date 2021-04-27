from src.utils.utils import shapes2arrays, increase_last_dim


class StateStacker:
    def __init__(self, state_shapes, stack_size, num_par_inst):
        self.state_shapes = state_shapes
        self.stack_shapes = increase_last_dim(state_shapes, factor=stack_size)
        self.stacks = shapes2arrays(self.stack_shapes, preceded_by=(num_par_inst,))

    def get_stack_shapes(self):
        return self.stack_shapes

    def add_states(self, states):
        shift_back(self.stacks)
        for i in range(len(self.stack_shapes)):
            self.stacks[i][..., -self.state_shapes[i][-1]:] = states[i]

    def get_stacks(self):
        return self.stacks

    def reset_stacks(self, ids):
        for stack_state_comp in self.stacks:
            stack_state_comp[ids] = 0


def shift_back(stacks):
    for stack_state_comp in stacks:
        stack_state_comp[..., :-1] = stack_state_comp[..., 1:]
