import numpy as np
from src.utils.utils import del_first
from src.utils.text_sty import print_error, yellow, orange


class ReplayMemory:
    def __init__(self, memory_size, state_shape, n_step, hidden_state_shape=None,
                 sequence_len=None, sequence_shift=None, eta=0.9):
        """A finite buffer for saving and sampling transitions.

        :param memory_size: the number of transitions the ReplayMemory can hold at most
        :param state_shape: list of 2d and 1d state dimensions
        """

        self.memory_size = memory_size
        self.n_step = n_step

        self.stack_ptr = 0
        self.seq_ptr = 0
        self.state_shape_2d = state_shape[0]
        self.state_shape_1d = state_shape[1]
        self.hidden_state_shape = hidden_state_shape
        self.sequential = hidden_state_shape is not None

        self.states_2d = np.zeros(shape=np.append([self.memory_size], self.state_shape_2d), dtype='float32')
        self.states_1d = np.zeros(shape=np.append([self.memory_size], self.state_shape_1d), dtype='float32')
        self.actions = np.zeros(shape=self.memory_size, dtype='int')
        self.scores = np.zeros(shape=self.memory_size, dtype='int')  # score *difference* between two transitions
        self.terminals = np.zeros(shape=self.memory_size, dtype='bool')
        self.priorities = np.zeros(shape=self.memory_size, dtype='float32')
        self.rewards = np.zeros(shape=self.memory_size, dtype='float32')
        self.returns = np.zeros(shape=self.memory_size, dtype='float32')  # discounted Monte Carlo returns

        if self.sequential:
            self.hidden_states = np.zeros(shape=np.append([self.memory_size], self.hidden_state_shape), dtype='float32')
            assert sequence_len is not None
            assert sequence_shift is not None
            assert sequence_shift <= sequence_len
            self.sequence_len = sequence_len
            self.sequence_shift = sequence_shift
            self.sequence_overlap = sequence_len - sequence_shift
            self.sequences = np.zeros(shape=(self.memory_size, 2), dtype='int')  # seq start, seq len
            self.sequence_priorities = np.zeros(shape=self.memory_size, dtype='float32')
        else:
            self.hidden_states = None
            self.sequence_len = None
            self.sequence_shift = None
            self.sequence_overlap = None
            self.sequences = None
            self.sequence_priorities = None

        self.eta = eta

        self.max_priority = 1

    def memorize(self, states, hidden_states, actions, scores, rewards, gamma):
        """Saves the observations of a whole episode."""

        # Retrieve episode length and determine episode end pointer
        ep_len = len(rewards)
        end_ptr = self.stack_ptr + ep_len

        remaining_space = self.memory_size - self.stack_ptr
        if ep_len > remaining_space:
            print(orange("WARNING: Episode too large to save. Episode has length %d but memory has"
                         "\n         remaining space for only %d transitions. Dropping this Episode." %
                         (ep_len, remaining_space)))
            return

        # Save observed data
        states_2d, states_1d = states
        self.states_2d[self.stack_ptr:end_ptr] = states_2d
        self.states_1d[self.stack_ptr:end_ptr] = states_1d
        if self.sequential:
            self.hidden_states[self.stack_ptr:end_ptr] = hidden_states
        self.actions[self.stack_ptr:end_ptr] = actions
        self.rewards[self.stack_ptr:end_ptr] = rewards
        self.scores[self.stack_ptr:end_ptr] = scores
        self.terminals[end_ptr - 1] = True  # Assuming unused part of self.terminals is kept False
        self.priorities[self.stack_ptr:end_ptr] = self.max_priority
        self.returns[end_ptr - 1] = rewards[-1]
        for i in range(1, ep_len):
            self.returns[end_ptr - i - 1] = rewards[-i - 1] + gamma * self.returns[end_ptr - i]

        # Add sequences to sequence list
        if self.sequential:
            seq_pos = 0
            while seq_pos < ep_len:
                seq_start = self.stack_ptr + seq_pos
                seq_len = min(self.sequence_len, ep_len - seq_pos)
                self.sequences[self.seq_ptr] = [seq_start, seq_len]
                self.sequence_priorities[self.seq_ptr] = self.max_priority
                seq_pos += self.sequence_shift
                self.seq_ptr += 1

        self.stack_ptr = end_ptr

        if self.stack_ptr / self.memory_size > 0.98:
            print(yellow("Info: Memory is running out of space! Only %.1f %% (%d transitions) left!" %
                         (100 - self.stack_ptr / self.memory_size * 100, remaining_space)))

    def recall_single_transitions(self, num_transitions, alpha):
        """Returns a batch of transition IDs, depending on the transitions' priorities.
        This is part of Prioritized Experience Replay."""

        sample_size = min(num_transitions, self.get_length())
        priorities = self.get_priorities()
        return choice_by_priority(sample_size, priorities, alpha)

    def recall_sequences(self, num_sequences, alpha, batch_size=None):
        """Returns a batch of sequence IDs, drawn according to priority. If batch_size is given, ensures
        that number of sequence ids returned is divisible by batch_size."""

        assert self.sequential

        num_sequences = min(num_sequences, self.get_num_sequences())
        priorities = self.get_sequence_priorities()
        seq_ids, probabilities = choice_by_priority(num_sequences, priorities, alpha)
        if batch_size is not None:
            overhang = len(seq_ids) % batch_size
            if overhang:
                seq_ids = seq_ids[:-overhang]

        return seq_ids, probabilities

    def get_length(self):
        return self.stack_ptr

    def get_num_sequences(self):
        if self.sequential:
            return self.seq_ptr
        else:
            return None

    def get_number_of_finished_episodes(self):
        return np.sum(self.get_terminals())

    def get_states(self):
        return self.states_2d[:self.stack_ptr], self.states_1d[:self.stack_ptr]

    def get_actions(self):
        return self.actions[:self.stack_ptr]

    def get_rewards(self):
        return self.rewards[:self.stack_ptr]

    def get_returns(self):
        return self.returns[:self.stack_ptr]

    def get_scores(self):
        return self.scores[:self.stack_ptr]

    def get_terminals(self):
        return self.terminals[:self.stack_ptr]

    def get_sequence_priorities(self):
        return self.sequence_priorities[:self.seq_ptr]

    def get_states_at(self, ids):
        """Returns an array, containing the selected experienced states."""
        return self.get_states()[ids]

    def get_transitions(self, trans_ids, mask=None):
        """Returns views of transitions, selected by their given IDs."""
        if np.max(trans_ids) >= self.stack_ptr:
            raise ValueError("Invalid transition indices given. You tried to read from memory "
                             "position %d which is not part of the active memory, as the top stack "
                             "pointer is at position %d." % (np.max(trans_ids), self.stack_ptr))

        states_2d = self.states_2d[trans_ids].copy()
        states_1d = self.states_1d[trans_ids].copy()
        if mask is not None:
            states_2d[~ mask] = 0
            states_1d[~ mask] = 0
        states = [states_2d, states_1d]

        if self.sequential:
            first_hidden_states = self.hidden_states[trans_ids[:, 0]]
        else:
            first_hidden_states = None

        actions = self.actions[trans_ids]
        terminals = self.terminals[trans_ids]

        # Obtain n-step rewards TODO: implement for sequential
        step_axis = trans_ids.ndim
        ids_repeated = np.repeat(np.expand_dims(trans_ids, axis=step_axis), self.n_step, axis=step_axis)
        lookaheads = np.mgrid[0:len(trans_ids), 0:self.n_step][1]
        n_step_trans_ids = ids_repeated + lookaheads
        n_step_mask = n_step_trans_ids < self.stack_ptr
        for step in range(1, self.n_step):
            step_trans_ids = trans_ids + step - 1
            step_trans_ids[step_trans_ids >= self.stack_ptr] = 0
            n_step_mask[:, step] = n_step_mask[:, step-1] & ~ self.terminals[step_trans_ids]
        n_step_trans_ids[~ n_step_mask] = 0

        n_step_rewards = self.rewards[n_step_trans_ids]

        next_trans_ids = np.copy(trans_ids) + self.n_step
        next_trans_ids[next_trans_ids >= self.stack_ptr] = 0
        # next_trans_ids[~ terminals] += self.n_step
        next_states_2d = self.states_2d[next_trans_ids]
        next_states_1d = self.states_1d[next_trans_ids]
        if mask is not None:
            next_states_2d[~ mask] = 0
            next_states_1d[~ mask] = 0
        next_states = [next_states_2d, next_states_1d]

        if self.sequential:
            last_hidden_states = self.hidden_states[next_trans_ids[:, -1]]
        else:
            last_hidden_states = None

        return states, first_hidden_states, actions, n_step_rewards, n_step_mask, \
               next_states, last_hidden_states, terminals

    def get_sequences(self, seq_ids):
        """Returns (zero-padded) sequences of transitions."""
        trans_ids, mask = self.seq_ids_to_trans_ids(seq_ids)
        return trans_ids, self.get_transitions(trans_ids, mask), mask

    def get_mc_returns(self, trans_ids):
        """Returns a list of Monte Carlo returns for a given list of transition IDs."""
        return self.get_returns()[trans_ids]

    def get_average_episode_length_for_last(self, num_episodes):
        if self.get_length() > 0:
            episode_endings = np.where(self.get_terminals())[0]
            if len(episode_endings) > num_episodes:
                final_n_endings = episode_endings[- num_episodes - 1:]
                total_length = final_n_endings[-1] - final_n_endings[0]
                avg_length = total_length / num_episodes
                return avg_length
            else:
                return 0
        else:
            return 0

    def get_priorities(self):
        return self.priorities[:self.stack_ptr]

    def set_priorities(self, trans_ids, priorities):
        self.priorities[trans_ids] = priorities
        self.max_priority = np.max((np.max(priorities), self.max_priority))

    def seq_ids_to_trans_ids(self, seq_ids):
        num_seqs = len(seq_ids)

        seq_starts = self.sequences[seq_ids, 0]
        seq_lengths = self.sequences[seq_ids, 1]
        seq_max_ends = seq_starts + self.sequence_len

        trans_ids = np.stack([np.arange(start, end) for start, end in zip(seq_starts, seq_max_ends)], axis=0)
        trans_ids[trans_ids >= self.get_length()] = 0  # are masked away anyway, but ids need to be legal indices

        # Mask sequences accordingly
        mask = np.zeros(shape=(num_seqs, self.sequence_len), dtype='bool')
        for seq in range(num_seqs):
            mask[seq, 0:seq_lengths[seq]] = True

        return trans_ids, mask

    def update_seq_priorities(self, seq_ids, trans_ids=None, mask=None):
        if trans_ids is None:
            trans_ids, mask = self.seq_ids_to_trans_ids(seq_ids)
        priorities = self.get_priorities()[trans_ids].copy()
        priorities[~ mask] = 0
        prio_max = np.max(priorities, axis=1)
        prio_avg = np.average(priorities, axis=1)
        seq_prios = self.eta * prio_max + (1 - self.eta) * prio_avg
        self.sequence_priorities[seq_ids] = seq_prios
        return seq_prios

    def reset_priorities(self):
        self.priorities[:] = 1
        if self.sequential:
            self.sequence_priorities[:] = 1

    def print_trans_from(self, trans_ids, env):
        for idx in trans_ids:
            print(self.get_trans_text(idx, env))

    def get_trans_text(self, idx, env):
        action_names = env.actions
        text = "\nTransition %d:\n" % idx + \
               env.state_2d_to_text(self.states_2d[idx]) + "\n" + \
               env.state_1d_to_text(self.states_1d[idx]) + \
               "\nAction:     " + str(action_names[self.actions[idx]]) + \
               "\nReward:     " + str(self.rewards[idx]) + \
               "\nScore gain: " + str(self.scores[idx]) + \
               "\nTerminal:   " + str(self.terminals[idx]) + \
               "\nPriority:   " + str(self.priorities[idx])
        return text

    def print_entire_memory(self, env):
        if self.get_length() > 1000000:
            print("Don't!")
            return
        self.print_trans_from(range(self.stack_ptr), env)

    def delete_first(self, n):
        """Deletes the first n transitions from this memory. Keeps the data in-place."""
        assert n <= self.stack_ptr

        for lst in [self.states_2d, self.states_1d, self.actions, self.scores, self.terminals, self.priorities,
                    self.rewards, self.returns]:
            del_first(lst, n)

        if self.sequential:
            del_first(self.hidden_states, n)
            k = np.argmax(self.sequences[:, 0] >= n)
            del_first(self.sequences, k)
            self.sequences[:, 0] -= n
            del_first(self.sequence_priorities, k)
            self.seq_ptr -= k

        self.stack_ptr -= n

    def get_state_shapes(self):
        return self.state_shape_2d, self.state_shape_1d

    def get_config(self):
        config = {"memory_size": self.memory_size,
                  "state_shape": [self.state_shape_2d, self.state_shape_1d],
                  "n_step": self.n_step,
                  "hidden_state_shape": self.hidden_state_shape,
                  "sequence_len": self.sequence_len,
                  "sequence_shift": self.sequence_shift,
                  "eta": self.eta}
        return config


def choice_by_priority(num_instances, priorities, alpha):
    """Chooses a sample of ids from a given priority list."""

    pool_size = len(priorities)
    assert pool_size >= num_instances

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
