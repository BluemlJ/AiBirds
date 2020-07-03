import numpy as np
import dask.array as da
import h5py
import os


class ReplayMemory:
    def __init__(self, state_res_per_dim, experience_path="data/experiences.hdf5", overwrite=False):
        # Initialize the experience, containing all (s, a, r, s', t) tuples experienced so far
        self.states = da.empty((0, state_res_per_dim, state_res_per_dim, 3), type=np.uint)
        self.actions = np.empty((0,), dtype='int')
        self.rewards = np.empty((0,), dtype='float32')
        self.terminals = np.empty((0,), dtype='bool')
        self.priorities = np.empty((0,), dtype='float32')

        self.state_res_per_dim = state_res_per_dim

        # Path string to location for saving the experience data
        self.experience_path = experience_path

        # Load existing experience if override is false (and experience exists)
        if not overwrite:
            if os.path.exists(experience_path):
                print("Reloading existing experience.")
                self.import_experience()
            else:
                print("No previous experience found at '%s'. A new experience dataset will be created." %
                      experience_path)
        else:
            if os.path.exists(experience_path):
                print("Overriding previously saved experience at %s." % experience_path)
                os.remove(experience_path)

        self.open_file = None

    def memorize(self, observations):
        obs_states_np = np.stack(observations[:, 0]).reshape((-1, 124, 124, 3))
        obs_states = da.from_array(obs_states_np)
        obs_actions = np.asarray(observations[:, 1], dtype=np.uint)
        obs_rewards = np.asarray(observations[:, 2], dtype='float32')
        terminals = np.array((len(observations) - 1) * [False] + [True], dtype='bool')
        max_priority = np.amax(self.get_priorities(), initial=1.0)
        priorities = np.asarray(len(observations) * [max_priority], dtype='float32')

        self.states = da.concatenate([self.states, obs_states], axis=0)
        self.actions = np.concatenate((self.actions, obs_actions), axis=0)
        self.rewards = np.concatenate((self.rewards, obs_rewards), axis=0)
        self.terminals = np.concatenate((self.terminals, terminals), axis=0)
        self.priorities = np.concatenate((self.priorities, priorities), axis=0)

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
        return self.states.shape[0]

    def get_transitions(self, trans_ids):
        """Returns an np.array of transitions, selected by their given IDs."""

        num_trans = len(trans_ids)

        states = self.states[trans_ids].compute().reshape((-1, 1, self.state_res_per_dim, self.state_res_per_dim, 3))
        actions = self.actions[trans_ids]
        rewards = self.rewards[trans_ids]
        terminals = self.terminals[trans_ids]
        next_states = np.zeros((num_trans, self.state_res_per_dim, self.state_res_per_dim, 3))

        # If terminal == True, next_state remains zero matrix
        next_states[terminals == False] = self.states[trans_ids[terminals == False] + 1]
        next_states = next_states.reshape((-1, 1, self.state_res_per_dim, self.state_res_per_dim, 3))

        transitions = list(zip(states, actions, rewards, next_states, terminals))
        return transitions

    def get_priorities(self):
        return self.priorities

    def set_priority(self, trans_id, priority):
        self.priorities[trans_id] = priority

    def reset_priorities(self):
        self.priorities = 1

    def export_experience(self, experience_path=None, overwrite=False):
        print("Exporting %d transitions..." % self.get_length())

        if experience_path is None:
            experience_path = self.experience_path

        if overwrite and os.path.exists(experience_path):
            if self.open_file is not None:
                self.open_file.close()
                self.open_file = None
            os.remove(experience_path)

        while os.path.exists(experience_path):
            print("There is already an experience dataset at '%s'." % experience_path)
            print("Please enter a different path:")
            experience_path = input()

        actions = da.from_array(self.actions, chunks=1)
        rewards = da.from_array(self.rewards, chunks=1)
        terminals = da.from_array(self.terminals, chunks=1)
        priorities = da.from_array(self.priorities, chunks=1)

        dataset = {"states": self.states,
                   "actions": actions,
                   "rewards": rewards,
                   "terminals": terminals,
                   "priorities": priorities}

        da.to_hdf5(experience_path, dataset, compression="gzip", compression_opts=6)

        print("Export finished.")

    def import_experience(self, experience_path=None):
        if experience_path is None:
            experience_path = self.experience_path

        print("Importing transitions from '%s'..." % experience_path)

        f = h5py.File(experience_path)
        states = da.from_array(f['states'])
        self.states = np.zeros((len(states), self.state_res_per_dim, self.state_res_per_dim, 3), dtype='float32')
        da.store(states, self.states)
        self.actions = f['actions'].value
        self.rewards = f['rewards'].value
        self.terminals = f['terminals'].value
        self.priorities = f['priorities'].value

        self.open_file = f

        print("Imported %d transitions." % self.get_length())
