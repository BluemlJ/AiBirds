import numpy as np
from src.mem.trans_buf import TransitionsBuffer
from src.mem.seq_mngr import SequenceManager
from src.utils.text_sty import print_error


class ReplayMemory:
    def __init__(self, size, state_shapes, state_dtypes, n_step, num_par_envs, stack_size=1,
                 hidden_state_shapes=None, sequence_len=None, sequence_shift=None, eta=0.9):
        """A finite buffer for saving and sampling transitions.

        :param size: the number of transitions the ReplayMemory can hold at most
        :param state_shapes: list of tuples of integers
        """

        self.n_step = n_step
        self.stack_size = stack_size  # frame stacking

        self.sequential = hidden_state_shapes is not None
        if self.sequential:
            assert sequence_len is not None
            assert sequence_shift is not None and sequence_shift <= sequence_len

        self.state_shapes = state_shapes
        self.state_dtypes = state_dtypes
        self.hidden_state_shapes = hidden_state_shapes

        self.trans_buf = TransitionsBuffer(size=size, state_shapes=state_shapes, state_dtypes=state_dtypes,
                                           hidden_shapes=hidden_state_shapes, par_inst=num_par_envs)
        if self.sequential:
            self.seq_mngr = SequenceManager(size, num_par_envs, sequence_len, sequence_shift, eta)
        else:
            self.seq_mngr = None

        self.max_priority = 1

    def memorize_observations(self, states, hidden_states, actions, scores, rewards, terminals, gamma):
        """Takes observations of one time-step from all parallel environments simultaneously. Returns
        the number of new transitions in this ReplayMemory available for learning."""

        self.trans_buf.save_transitions(states, hidden_states, actions, scores, rewards, terminals, gamma)
        if self.sequential:
            self.seq_mngr.update(terminals, self.trans_buf.stack_ptr, self.max_priority)

        return len(actions)

    def recall(self, num_instances, alpha, batch_size=None):
        """Returns a batch of transition IDs if mode is non-sequential, else a batch of sequence IDs.
        Choice is made using the principle of prioritized experience replay."""
        if not self.sequential:
            priorities = self.get_priorities()[:-self.n_step].flatten()
            sample_size = min(num_instances, self.get_num_learnable_transitions())
            return choice_by_priority(sample_size, priorities, alpha)
        else:
            priorities = self.get_priorities()
            num_sequences = min(num_instances, self.get_num_sequences())
            seq_ids, probabilities = choice_by_priority(num_sequences, priorities, alpha)
            if batch_size is not None:
                overhang = len(seq_ids) % batch_size
                if overhang:
                    seq_ids = seq_ids[:-overhang]
            return seq_ids, probabilities

    def get_size(self):
        return self.trans_buf.buffer_len * self.trans_buf.num_par_inst

    def get_num_transitions(self):
        return self.trans_buf.get_num_transitions()

    def get_num_learnable_transitions(self):
        """Returns the number of transitions in this memory which are allowed to be used
        for training. In particular, excludes incomplete n-steps."""
        trans_per_env = self.trans_buf.stack_ptr - self.n_step
        total_trans = trans_per_env * self.trans_buf.num_par_inst
        return max(0, total_trans)

    def get_num_sequences(self):
        if self.sequential:
            return self.seq_mngr.get_num_seqs()
        else:
            return None

    def get_states(self, trans_ids=None):
        """Returns observed as well as hidden states"""
        if trans_ids is None:
            return self.trans_buf.get_states()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_states(trans_indices)

    def get_actions(self, trans_ids=None):
        if trans_ids is None:
            return self.trans_buf.get_actions()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_actions(trans_indices)

    def get_rewards(self, trans_ids=None):
        if trans_ids is None:
            return self.trans_buf.get_rewards()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_rewards(trans_indices)

    def get_scores(self, trans_ids=None):
        if trans_ids is None:
            return self.trans_buf.get_scores()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_scores(trans_indices)

    def get_returns(self, trans_ids=None):
        if trans_ids is None:
            return self.trans_buf.get_returns()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_returns(trans_indices)

    def get_terminals(self, trans_ids=None):
        if trans_ids is None:
            return self.trans_buf.get_terminals()
        else:
            trans_indices = self.id2idx(trans_ids)
            return self.trans_buf.get_terminals(trans_indices)

    def get_priorities(self, inst_ids=None):
        if inst_ids is None:
            return self.trans_buf.get_priorities() if not self.sequential else self.seq_mngr.get_seq_prios()
        else:
            if not self.sequential:
                trans_indices = self.id2idx(inst_ids)
                return self.trans_buf.get_priorities(trans_indices)
            else:
                return self.seq_mngr.get_seq_prios()[inst_ids]

    def get_transitions(self, trans_ids):
        """Returns views of transitions, selected by their given IDs (not indices!)."""
        trans_indices = self.id2idx(trans_ids)
        return self.trans_buf.get_transitions(trans_indices, self.n_step, self.stack_size)

    def get_sequences(self, seq_ids):
        """Returns sequences of transitions."""
        trans_ids, mask = self.seq_mngr.seq_ids_to_trans_ids(seq_ids)
        return trans_ids, mask, self.get_transitions(trans_ids)

    def get_mc_returns(self, trans_ids):
        """Returns a list of Monte Carlo returns for a given list of transition IDs."""
        trans_indices = self.id2idx(trans_ids)
        return self.get_returns()[trans_indices[0], trans_indices[1]]

    def set_priorities(self, trans_ids, priorities):
        trans_indices = self.id2idx(trans_ids)
        self.trans_buf.set_priorities(trans_indices, priorities)

    def print_trans_from(self, trans_id, env):
        print(self.get_trans_text(trans_id, env))

    def get_trans_text(self, trans_id, env):
        action_names = env.actions
        state = self.get_states(trans_id)[0]
        text = "\nTransition %d:\n" % trans_id + env.state2text(state) + \
               "\nAction:     " + str(action_names[self.get_actions(trans_id)]) + \
               "\nReward:     " + str(self.get_rewards(trans_id)) + \
               "\nScore:      " + str(self.get_scores(trans_id)) + \
               "\nTerminal:   " + str(self.get_terminals(trans_id)) + \
               "\nPriority:   %.2f" % self.get_priorities(trans_id)
        return text

    def print_entire_memory(self, env):
        if self.get_num_transitions() > 1000000:
            print("Don't!")
            return
        for trans_id in range(self.trans_buf.get_num_transitions()):
            self.print_trans_from(trans_id, env)

    def delete_first(self, n):
        """Deletes the first (about) n transitions from this memory."""
        steps_to_del = n // self.trans_buf.num_par_inst

        # Delete transitions (and sequences)
        self.trans_buf.delete_first(steps_to_del)
        if self.sequential:
            self.seq_mngr.delete_first(steps_to_del)

    def get_config(self):
        config = {"size": self.trans_buf.size,
                  "state_shapes": self.state_shapes,
                  "n_step": self.n_step,
                  "num_par_envs": self.trans_buf.num_par_inst,
                  "hidden_state_shapes": self.hidden_state_shapes,
                  "sequence_len": self.seq_mngr.seq_len if self.sequential else None,
                  "sequence_shift": self.seq_mngr.seq_shift if self.sequential else None,
                  "eta": self.seq_mngr.eta if self.sequential else None}
        return config

    def id2idx(self, ids):
        step_indices = ids // self.trans_buf.num_par_inst
        env_indices = ids % self.trans_buf.num_par_inst
        return np.array([step_indices, env_indices])

    def idx2id(self, indices):
        step_indices, env_indices = indices
        ids = step_indices * self.trans_buf.num_par_inst + env_indices
        return ids


def choice_by_priority(num_instances, priorities, alpha):
    """Chooses a sample of ids from a given priority list."""
    pool_size = len(priorities)
    assert pool_size >= num_instances

    if num_instances == 0:
        return [], []

    # Take power of each element with alpha to adjust priorities
    adjusted_priorities = np.power(priorities, alpha)

    total_prio = np.sum(adjusted_priorities)
    if total_prio == 0:  # Catch
        print_error("Error: All given priorities are zero! Since this is practically impossible, "
                    "something might be wrong.\nThis is no critical error, hence, training continues.")
        return [], []

    # Convert priorities into probabilities
    probabilities = adjusted_priorities / np.sum(adjusted_priorities)

    # Handle cases with less non-zero probabilities than sample_size
    num_instances = np.min((np.count_nonzero(probabilities), num_instances))

    # Randomly select transitions with given probabilities
    instance_ids = np.random.choice(range(pool_size), size=num_instances, p=probabilities, replace=False)

    return instance_ids, probabilities
