import numpy as np
from src.utils.utils import shapes2arrays
from src.utils.array_ops import del_first


class TransitionsBuffer:
    """Saves transitions in a parallel manner (somewhat like a "block buffer").
    Axes:
    0: step_id
    1: env_id
    2-n: scalar values or state shape"""

    ACTIONS = 0
    SCORES = 1
    REWARDS = 2
    RETURNS = 3  # TODO: not allowed in instant learn mode
    TERMINALS = 4
    PRIORITIES = 5

    def __init__(self, size, state_shapes, state_dtypes, hidden_shapes, par_inst):
        self.buffer_len = size // par_inst
        self.size = self.buffer_len * par_inst  # number of transitions in total
        self.stack_ptr = 0
        self.saving_hidden_states = hidden_shapes is not None

        self.num_par_inst = par_inst
        self.state_shapes = state_shapes
        self.state_dtypes = state_dtypes
        self.hidden_state_shapes = hidden_shapes

        self.scalar_obs = np.zeros(shape=(self.buffer_len, self.num_par_inst, 6), dtype="float32")

        self.print_mem_usage()
        self.states = shapes2arrays(state_shapes, preceded_by=(self.buffer_len, self.num_par_inst),
                                    dtypes=self.state_dtypes)
        if self.saving_hidden_states:
            self.hidden_states = shapes2arrays(hidden_shapes, preceded_by=(self.buffer_len, self.num_par_inst),
                                               dtypes=self.state_dtypes)
        else:
            self.hidden_states = None

        self.initial_priority = 1

    def save_transitions(self, states, hidden_states, actions, scores, rewards, terminals, gamma):
        self.save_states(states, hidden_states)
        self.scalar_obs[self.stack_ptr, :, self.ACTIONS] = actions
        self.scalar_obs[self.stack_ptr, :, self.SCORES] = scores
        self.scalar_obs[self.stack_ptr, :, self.REWARDS] = rewards
        self.scalar_obs[self.stack_ptr, :, self.TERMINALS] = terminals
        self.scalar_obs[self.stack_ptr, :, self.PRIORITIES] = self.initial_priority

        self.stack_ptr += 1

        if np.any(terminals):
            self.compute_returns_for(terminals, gamma)

    def compute_returns_for(self, env_ids, gamma):
        comp_return = env_ids.copy()
        trans_ptr = 1
        while np.any(comp_return):
            self.scalar_obs[self.stack_ptr - trans_ptr, comp_return, self.RETURNS] = \
                self.scalar_obs[self.stack_ptr - trans_ptr, comp_return, self.REWARDS] + gamma * \
                self.scalar_obs[self.stack_ptr - trans_ptr + 1, comp_return, self.RETURNS]
            trans_ptr += 1
            if trans_ptr > self.stack_ptr:
                return
            comp_return &= ~self.scalar_obs[self.stack_ptr - trans_ptr, :, self.TERMINALS].astype("bool")

    def save_states(self, states, hidden_states):
        for state_comps, state_comp in zip(self.states, states):
            state_comps[self.stack_ptr] = state_comp
        if self.saving_hidden_states:
            for hidden_comps, hidden_comp in zip(self.hidden_states, hidden_states):
                hidden_comps[self.stack_ptr] = hidden_comp

    def get_states(self, trans_indices=None, stack_size=1):
        assert trans_indices is not None or stack_size == 1, NotImplementedError

        states = []
        hidden_states = []

        if trans_indices is None:
            for state_comps in self.states:
                states += [state_comps[:self.stack_ptr]]
            if self.saving_hidden_states:
                for hidden_comps in self.hidden_states:
                    hidden_states += [hidden_comps[:self.stack_ptr]]

        else:
            # Prepare indices for frame-stacking
            step_indices, env_indices = trans_indices
            lookbacks = np.zeros(shape=(*step_indices.shape, stack_size), dtype="int")
            valid = np.ones(step_indices.shape, dtype="bool")
            for frame in range(1, stack_size):
                valid &= (step_indices - frame >= 0) & \
                         ~ self.get_terminals([step_indices - frame, env_indices])
                lookbacks[..., stack_size - 1 - frame] = lookbacks[..., stack_size - frame] - valid

            # Transform single-frame indices to stacked-frame indices
            step_stack_indices = np.repeat(np.expand_dims(step_indices, axis=-1), stack_size, axis=-1) + lookbacks
            env_stack_indices = np.repeat(np.expand_dims(env_indices, axis=-1), stack_size, axis=-1)

            # Extract from state components
            states = get_stack_state_comp(self.states, step_stack_indices, env_stack_indices)
            if self.saving_hidden_states:
                hidden_states = get_stack_state_comp(self.hidden_states, step_stack_indices,
                                                     env_stack_indices)

        return states, hidden_states

    def del_states(self, n):
        for state_comps in self.states:
            del_first(state_comps, n)
        if self.saving_hidden_states:
            for hidden_comps in self.hidden_states:
                del_first(hidden_comps, n)

    def get_actions(self, trans_indices=None):
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.ACTIONS].astype("int")
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.ACTIONS].astype("int")

    def get_rewards(self, trans_indices=None):
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.REWARDS]
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.REWARDS]

    def get_scores(self, trans_indices=None):
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.SCORES]
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.SCORES]

    def get_returns(self, trans_indices=None):  # TODO: contains invalid returns
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.RETURNS]
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.RETURNS]

    def get_terminals(self, trans_indices=None):
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.TERMINALS].astype("bool")
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.TERMINALS].astype("bool")

    def get_priorities(self, trans_indices=None):
        if trans_indices is None:
            return self.scalar_obs[:self.stack_ptr, :, self.PRIORITIES]
        else:
            step_indices, env_indices = trans_indices
            return self.scalar_obs[step_indices, env_indices, self.PRIORITIES]

    def get_transitions(self, trans_indices, steps, stack_size):  # TODO: implement for sequential
        """
        :param trans_indices: list of lists of form [buff_pos, env_id], where request_shape := buff_pos.shape
        :param steps: n of n-steps
        :param stack_size:
        :return:
        - states: stacked states of shape (*request_shape, *state_shape, stack_size)
        - hidden_states: if sequential mode, model's hidden states of first step, has shape
           (*request_shape, *hidden_shape)
        - actions: actions taken in first step, has shape request_shape
        - n_step_rewards: rewards gained for each step, shape (*request_shape, n)
        - step_mask: Boolean mask of shape (*request_shape, n) indicating valid steps
        - next_states: stacked next (n-th) states of shape (*request_shape, *state_shape, stack_size)
        - next_hidden_states: if sequential mode, model's hidden states of next (n-th) step, has shape
           (*request_shape, *hidden_shape)
        """
        step_indices, env_indices = trans_indices
        if np.any(step_indices > self.stack_ptr - steps) or np.any(env_indices >= self.num_par_inst):
            raise ValueError("Invalid transition indices! Given indices lie outside of the "
                             "transitions buffer or the accessed transitions are not available "
                             "for learning yet.")
        num_trans = len(step_indices)

        states, hidden_states = self.get_states(trans_indices, stack_size)

        actions = self.get_actions(trans_indices)

        # Obtain n-step rewards
        lookahead_axis = step_indices.ndim
        step_indices_repeated = np.repeat(np.expand_dims(step_indices, axis=lookahead_axis), steps, axis=lookahead_axis)
        env_indices_repeated = np.repeat(np.expand_dims(env_indices, axis=lookahead_axis), steps, axis=lookahead_axis)
        lookaheads = np.mgrid[0:num_trans, 0:steps][1]
        n_step_indices = step_indices_repeated + lookaheads
        n_step_rewards = self.scalar_obs[n_step_indices, env_indices_repeated, self.REWARDS]

        # Build mask to accommodate n-step sequences which terminate during next n steps
        step_mask = np.ones(shape=(*n_step_indices.shape[:-1], steps + 1), dtype="bool")
        for k in range(1, steps + 1):
            lookahead_indices = step_indices + k - 1
            step_mask[:, k] = step_mask[:, k - 1] \
                              & ~ self.scalar_obs[lookahead_indices, env_indices, self.TERMINALS].astype("bool")

        next_step_indices = np.copy(step_indices) + steps
        next_states, next_hidden_states = self.get_states([next_step_indices, env_indices], stack_size)

        return states, hidden_states, actions, n_step_rewards, step_mask, \
               next_states, next_hidden_states

    def get_num_transitions(self):
        return self.stack_ptr * self.num_par_inst

    def set_priorities(self, trans_indices, priorities):
        step_indices, env_indices = trans_indices
        self.scalar_obs[step_indices, env_indices, self.PRIORITIES] = priorities
        self.initial_priority = np.max((np.max(priorities), self.initial_priority))

    def delete_first(self, n):
        """Deletes the first n parallel steps of transitions from this memory. Keeps the data in-place."""
        assert n <= self.stack_ptr
        self.del_states(n)
        del_first(self.scalar_obs, n)
        self.stack_ptr -= n

    def print_mem_usage(self):
        bytes_per_trans = np.int64(0)
        for shape, dtype in zip(self.state_shapes, self.state_dtypes):
            bytes_per_trans += np.prod(shape) * dtype.itemsize
        if self.saving_hidden_states:
            for shape in self.hidden_state_shapes:
                bytes_per_trans += np.prod(shape) * self.hidden_states[0].itemsize
        bytes_per_trans += self.scalar_obs.shape[-1] * self.scalar_obs.itemsize
        total_bytes = bytes_per_trans * self.size
        gib = total_bytes / 2 ** 30
        print("Reserving %.2f GiB of RAM for TransitionsBuffer (ReplayMemory)..." % gib)


def get_stack_state_comp(states, step_stack_indices, env_stack_indices):
    extracted_states = []
    for state_comps in states:
        state_comp = state_comps[step_stack_indices, env_stack_indices]
        stacks_sep = np.moveaxis(state_comp, step_stack_indices.ndim - 1, -2)
        extracted_states += [np.reshape(stacks_sep, (*stacks_sep.shape[:-2], -1))]
    return extracted_states
