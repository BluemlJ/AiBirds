import numpy as np
import dask.array as da
import h5py
import os


class ReplayMemory:
    def __init__(self, state_res_per_dim, score_normalization, import_from=None):
        self.state_res_per_dim = state_res_per_dim
        self._initialize_experience()
        self.gamma = None  # discount factor used for returns list

        # State pixel resolution per dimension (width and height)
        self.state_res_per_dim = state_res_per_dim

        self.score_normalization = score_normalization

        # Currently opened data files (used by Dask for efficiency)
        self.open_files = []

        # Ignore Warnings caused by the Dask package
        np.seterr(divide='ignore', invalid='ignore')

        if import_from is not None:
            if os.path.exists(import_from):
                print("Reloading existing experience from '%s'." % import_from)
                self.import_experience(experience_path=import_from)
            else:
                print("No previous experience found at '%s'. A new experience dataset will be created"
                      "at this location." % import_from)

    def _initialize_experience(self):
        self.states = da.empty((0, 1, self.state_res_per_dim, self.state_res_per_dim, 3), dtype=np.uint8)
        self.new_states = np.empty((0, 1, self.state_res_per_dim, self.state_res_per_dim, 3), dtype=np.uint8)
        self.actions = np.empty((0,), dtype='int')
        self.scores = np.empty((0,), dtype='int')
        self.terminals = np.empty((0,), dtype='bool')
        self.won = np.empty((0,), dtype='bool')
        self.priorities = np.empty((0,), dtype='float32')
        self.rewards = np.empty((0,), dtype='float32')
        self.returns = np.empty((0,), dtype='float32')  # discounted Monte Carlo returns

    def memorize(self, observations, rewards, won, gamma, grace_factor):
        """Saves the observations of a whole episode.

        :param observations: list of (state, action, score)
        :param rewards:
        :param won: True if episode was completed successfully, False otherwise
        :param gamma: discount factor
        :param grace_factor:
        """

        episode_length = len(observations)

        # Update rewards if necessary
        if len(self.rewards) != self.get_length():
            self.calculate_rewards(grace_factor)

        # Update returns with new discount factor, if necessary
        if self.gamma != gamma:
            self.calculate_returns(gamma)

        obs_states = np.stack(observations[:, 0])
        obs_actions = np.asarray(observations[:, 1], dtype='int')
        obs_scores = np.asarray(observations[:, 2], dtype='int')
        terminals = np.array((episode_length - 1) * [False] + [True], dtype='bool')
        won = np.array(episode_length * [won], dtype='bool')
        max_priority = np.amax(self.get_priorities(), initial=1.0)
        priorities = np.asarray(episode_length * [max_priority], dtype='float32')
        returns = np.empty((episode_length,), dtype='float32')
        returns[-1] = rewards[-1]
        for i in reversed(range(episode_length - 1)):
            returns[i] = rewards[i] + gamma * returns[i + 1]

        self.new_states = np.concatenate((self.new_states, obs_states))
        self.actions = np.concatenate((self.actions, obs_actions))
        self.rewards = np.concatenate((self.rewards, rewards))
        self.scores = np.concatenate((self.scores, obs_scores))
        self.terminals = np.concatenate((self.terminals, terminals))
        self.won = np.concatenate((self.won, won))
        self.priorities = np.concatenate((self.priorities, priorities))
        self.returns = np.concatenate((self.returns, returns))

    def recall(self, num_transitions, alpha):
        """Returns a batch of transition IDs, depending on the transitions' priorities.
        This is part of Prioritized Experience Replay."""

        # Obtain number of experienced transitions
        exp_len = self.get_length()

        # Obtain batch size
        batch_size = np.min((exp_len, num_transitions))

        # Obtain priorities
        priorities = self.get_priorities()

        # Take power of each element with alpha to adjust priorities
        adjusted_priorities = np.power(priorities, alpha)

        # Convert priorities into probabilities
        probabilities = adjusted_priorities / np.sum(adjusted_priorities)

        # Randomly select transitions with given probabilities
        trans_ids = np.random.choice(range(exp_len), size=batch_size, p=probabilities)

        return trans_ids, probabilities

    def get_length(self):
        return self.won.shape[0]

    def get_states(self):
        """Returns an (uncomputed) Dask array, pointing to all experienced states."""
        return da.concatenate([self.states, self.new_states], axis=0)

    def get_transitions(self, trans_ids, grace_factor):
        """Returns an np.array of transitions, selected by their given IDs."""
        print("\033[92mFetching transitions...\033[0m")

        if len(self.rewards) != self.get_length():
            self.calculate_rewards(grace_factor)

        num_trans = len(trans_ids)

        states = self.get_states()[trans_ids].compute()
        actions = self.actions[trans_ids]
        rewards = self.rewards[trans_ids]
        terminals = self.terminals[trans_ids]
        next_states = np.zeros((num_trans, 1, self.state_res_per_dim, self.state_res_per_dim, 3))

        # If terminal == True, next_state remains zero matrix
        next_states[terminals == False] = self.get_states()[trans_ids[terminals == False] + 1].compute()

        print("\033[92mReturned %d transitions.\033[0m" % len(trans_ids))
        return states, actions, rewards, next_states, terminals

    def calculate_rewards(self, grace_factor):
        self.rewards = self.scores / self.score_normalization
        self.rewards[self.won == False] = self.rewards[self.won == False] * grace_factor

    def get_returns(self, trans_ids, gamma):
        """Returns a list of returns for a given list of transition IDs."""
        # Update returns list if necessary
        if self.gamma != gamma or len(self.returns) != self.get_length():
            self.calculate_returns(gamma)

        return self.returns[trans_ids]

    def calculate_returns(self, gamma):
        """Completely (re)calculates the returns list."""
        self.gamma = gamma
        self.returns = np.empty((self.get_length(),), dtype='float32')

        # Set return of all terminal transitions to their reward
        episode_iterators = np.where(self.terminals == True)[0]
        self.returns[episode_iterators] = self.rewards[episode_iterators]

        # Move iterators down by one transition
        episode_iterators -= 1

        # Remove all iterators which point at the last transition of the previous episode
        episode_iterators = episode_iterators[self.terminals[episode_iterators] == False]

        while len(episode_iterators) > 0:
            # Set returns inductively
            self.returns[episode_iterators] = self.rewards[episode_iterators] + \
                                            gamma * self.returns[episode_iterators + 1]

            episode_iterators -= 1
            episode_iterators = episode_iterators[self.terminals[episode_iterators] == False]

    def get_priorities(self):
        return self.priorities

    def set_priority(self, trans_id, priority):
        self.priorities[trans_id] = priority

    def set_priorities(self, trans_ids, priorities):
        self.priorities[trans_ids] = priorities

    def reset_priorities(self):
        self.priorities = np.ones((self.get_length(),), dtype='float32')

    def export_experience(self, experience_path, overwrite=False, compress=False):
        """Exports the current experience dataset (states, actions etc.). The exported data can be compressed
        with gzip via 'compress'."""

        while os.path.exists(experience_path) and not overwrite:
            print("There is already an experience dataset at '%s'." % experience_path)
            print("Please enter a different path:")
            experience_path = input()

        while not self.safe_to_write_at(experience_path):
            print("Warning! You are trying to overwrite a currently active experience data file. To avoid"
                  "memory issues, please specify a different export path:")
            experience_path = input()

        if overwrite and os.path.exists(experience_path):
            os.remove(experience_path)

        print("Exporting %d transitions..." % self.get_length())

        actions = da.from_array(self.actions, chunks=1)
        scores = da.from_array(self.scores, chunks=1)
        terminals = da.from_array(self.terminals, chunks=1)
        won = da.from_array(self.won, chunks=1)
        priorities = da.from_array(self.priorities, chunks=1)

        dataset = {"states": self.get_states(),
                   "actions": actions,
                   "scores": scores,
                   "terminals": terminals,
                   "won": won,
                   "priorities": priorities}

        if compress:
            da.to_hdf5(experience_path, dataset, compression="gzip", compression_opts=5)
        else:
            da.to_hdf5(experience_path, dataset)

        print("Export finished.")

    def import_experience(self, experience_path, grace_factor=None, gamma=None):
        print("Import started...")

        self.close_all_open_files()
        self._initialize_experience()
        self.add_experience(experience_path, grace_factor, gamma)

    def add_experience(self, experience_path, grace_factor=None, gamma=None):
        if not os.path.exists(experience_path):
            print("No experience found at '%s'. Continuing without adding experience." % experience_path)
            return

        print("Adding transitions from '%s'..." % experience_path)

        f = h5py.File(experience_path, mode="a")
        states_pointer = f['states']
        states = da.from_array(states_pointer, chunks=(1, 1, self.state_res_per_dim, self.state_res_per_dim, 3))
        actions = f['actions'][()]
        scores = f['scores'][()]
        terminals = f['terminals'][()]
        won = f['won'][()]
        priorities = f['priorities'][()]

        self.states = da.concatenate([self.states, states])
        self.actions = np.concatenate((self.actions, actions))
        self.scores = np.concatenate((self.scores, scores))
        self.terminals = np.concatenate((self.terminals, terminals))
        self.won = np.concatenate((self.won, won))
        self.priorities = np.concatenate((self.priorities, priorities))

        self.add_open_file(f)

        if grace_factor is not None:
            self.calculate_rewards(grace_factor)

        if gamma is not None:
            self.calculate_returns(gamma)

        print("Added %d transitions." % len(states))

    def add_open_file(self, f):
        self.open_files += [f]

    def close_all_open_files(self):
        for f in self.open_files:
            f.close()
        self.open_files = []

    def safe_to_write_at(self, path):
        for f in self.open_files:
            if path == f.filename:
                return False
        return True
