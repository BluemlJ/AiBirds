import numpy as np


class ReplayMemory:
    def __init__(self, memory_size, image_state_shape, numerical_state_shape):
        """A finite buffer for saving and sampling transitions.

        :param memory_size: the number of transitions the ReplayMemory can hold at most
        :param image_state_shape: list of image state dimensions
        :param numerical_state_shape: list of numerical state dimensions
        """

        self.memory_size = memory_size
        self.stack_ptr = 0
        self.image_state_shape = image_state_shape
        self.numerical_state_shape = numerical_state_shape

        # 2d data (plus 1 dimension for channels):
        self.image_states = np.zeros(shape=np.append([self.memory_size], self.image_state_shape), dtype='bool')
        # 1d data:
        self.numerical_states = np.zeros(shape=np.append([self.memory_size], self.numerical_state_shape),
                                         dtype='float32')
        self.actions = np.zeros(shape=self.memory_size, dtype='int')
        self.scores = np.zeros(shape=self.memory_size, dtype='int')  # score *difference* between two transitions
        self.terminals = np.zeros(shape=self.memory_size, dtype='bool')
        self.priorities = np.zeros(shape=self.memory_size, dtype='float32')
        self.rewards = np.zeros(shape=self.memory_size, dtype='float32')
        self.returns = np.zeros(shape=self.memory_size, dtype='float32')  # discounted Monte Carlo returns

        self.gamma = None  # discount factor used for return computation

        self.max_priority = 1

        # Currently opened data files (used by Dask for efficiency)
        self.open_files = []

    def memorize(self, obs_states, obs_actions, obs_scores, obs_rewards, gamma):
        """Saves the observations of a whole episode."""

        # Retrieve episode length and determine episode end pointer
        ep_len = len(obs_rewards)
        end_ptr = self.stack_ptr + ep_len

        remaining_space = self.memory_size - self.stack_ptr
        if ep_len > remaining_space:
            print("WARNING: Episode too large to save. Episode has length %d but memory has"
                  "\n         remaining space for only %d transitions. Dropping this Episode." %
                  (ep_len, remaining_space))
            return

        # Save observed data
        obs_state_images, obs_state_numerics = obs_states
        self.image_states[self.stack_ptr:end_ptr] = obs_state_images
        self.numerical_states[self.stack_ptr:end_ptr] = obs_state_numerics
        self.actions[self.stack_ptr:end_ptr] = obs_actions
        self.rewards[self.stack_ptr:end_ptr] = obs_rewards
        self.scores[self.stack_ptr:end_ptr] = obs_scores
        self.terminals[end_ptr - 1] = True  # Assuming unused part of self.terminals is kept False
        self.priorities[self.stack_ptr:end_ptr] = self.max_priority
        self.returns[end_ptr - 1] = obs_rewards[-1]
        for i in range(1, ep_len):
            self.returns[end_ptr - i - 1] = obs_rewards[-i] + gamma * self.returns[end_ptr - i]

        self.stack_ptr = end_ptr

        if self.stack_ptr / self.memory_size > 0.98:
            print("WARNING: Memory is running out of space! Only %.1f %% (%d transitions) left!" %
                  (100 - self.stack_ptr / self.memory_size * 100, remaining_space))

    def recall(self, num_transitions, alpha):
        """Returns a batch of transition IDs, depending on the transitions' priorities.
        This is part of Prioritized Experience Replay."""

        # Obtain number of experienced transitions
        exp_len = self.get_length()

        # Obtain sample size
        sample_size = np.min((exp_len, num_transitions))

        # Obtain priorities
        priorities = self.get_priorities()

        # Take power of each element with alpha to adjust priorities
        adjusted_priorities = np.power(priorities, alpha)

        # Convert priorities into probabilities
        probabilities = adjusted_priorities / np.sum(adjusted_priorities)

        # Handle cases with less non-zero probabilities than sample_size
        sample_size = np.min((np.count_nonzero(probabilities), sample_size))

        # Randomly select transitions with given probabilities
        trans_ids = np.random.choice(range(exp_len), size=sample_size, p=probabilities, replace=False)

        return trans_ids, probabilities

    def get_length(self):
        return self.stack_ptr

    def get_number_of_finished_episodes(self):
        return np.sum(self.get_terminals())

    def get_states(self):
        return self.image_states[:self.stack_ptr], self.numerical_states[:self.stack_ptr]

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

    def get_states_at(self, ids):
        """Returns an array, containing the selected experienced states."""
        return self.get_states()[ids]

    def get_transitions(self, trans_ids):
        """Returns views of transitions, selected by their given IDs."""
        if np.max(trans_ids) >= self.stack_ptr:
            raise ValueError("Invalid transition indices given. You tried to read from memory"
                             "position %d which is not part of the active memory as the top stack"
                             "pointer is at position %d." % (np.max(trans_ids), self.stack_ptr))

        state_images = self.image_states[trans_ids]
        state_numerics = self.numerical_states[trans_ids]
        states = [state_images, state_numerics]
        actions = self.actions[trans_ids]
        rewards = self.rewards[trans_ids]
        terminals = self.terminals[trans_ids]
        next_trans_ids = np.copy(trans_ids)
        next_trans_ids[~ terminals] += 1
        next_state_images = self.image_states[next_trans_ids]
        next_state_numerics = self.numerical_states[next_trans_ids]
        next_states = [next_state_images, next_state_numerics]

        return states, actions, rewards, next_states, terminals

    def calculate_rewards(self):
        # TODO: evaluate necessity of this function
        self.rewards = self.scores

    def get_mc_returns(self, trans_ids, gamma):
        """Returns a list of Monte Carlo returns for a given list of transition IDs."""
        # Update returns list if necessary
        """if self.gamma != gamma:
            self.calculate_returns(gamma)"""

        return self.get_returns()[trans_ids]

    def get_all_final_scores(self):
        self.calculate_rewards()
        self.calculate_returns(1)

        # Get the IDs of all transitions which are the first in their episode
        trans_ids = np.where(self.get_terminals())[0]
        trans_ids += 1
        trans_ids = trans_ids[:-1]
        trans_ids = np.insert(trans_ids, 0, 0)

        return self.returns[trans_ids]

    def calculate_returns(self, gamma):
        """Completely (re)calculates the returns list."""
        self.gamma = gamma
        self.returns[:] = 0

        # Set return of all terminal transitions to their reward
        episode_iterators = np.where(self.get_terminals())[0]
        self.get_returns()[episode_iterators] = self.get_rewards()[episode_iterators]

        # Move iterators down by one transition
        episode_iterators -= 1

        # Remove all iterators which point at the last transition of the previous episode
        episode_iterators = episode_iterators[~ self.get_terminals()[episode_iterators]]

        while len(episode_iterators) > 0:
            # Set returns inductively
            self.returns[episode_iterators] = self.rewards[episode_iterators] + \
                                              gamma * self.returns[episode_iterators + 1]

            episode_iterators -= 1
            episode_iterators = episode_iterators[~ self.get_terminals()[episode_iterators]]

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

    def reset_priorities(self):
        self.priorities = np.ones((self.get_length(),), dtype='float32')

    def print_trans_from(self, trans_ids, env):
        for idx in trans_ids:
            print(self.get_trans_text(idx, env))

    def get_trans_text(self, idx, env):
        action_names = env.actions
        text = "\nTransition %d:\n" % idx + \
               env.image_state_to_text(self.image_states[idx]) + "\n" + \
               env.numerical_state_to_text(self.numerical_states[idx]) + \
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

        m = self.memory_size - n

        self.image_states[:m] = self.image_states[n:]
        self.numerical_states[:m] = self.numerical_states[n:]
        self.actions[:m] = self.actions[n:]
        self.scores[:m] = self.scores[n:]
        self.terminals[:m] = self.terminals[n:]
        self.priorities[:m] = self.priorities[n:]
        self.rewards[:m] = self.rewards[n:]
        self.returns[:m] = self.returns[n:]

        self.image_states[-n:] = False
        self.numerical_states[-n:] = 0
        self.actions[-n:] = 0
        self.scores[-n:] = 0
        self.terminals[-n:] = False
        self.priorities[-n:] = 0
        self.rewards[-n:] = 0
        self.returns[-n:] = 0

        self.stack_ptr -= n

    def get_state_shapes(self):
        return self.image_state_shape, self.numerical_state_shape
