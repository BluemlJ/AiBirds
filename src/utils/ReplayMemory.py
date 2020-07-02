import numpy as np
import pickle
import bz2
import os


class ReplayMemory:
    def __init__(self, experience_path="data/experiences.bz2", override=False):
        # Initialize the experience list, a list of transitions
        # containing all (s, a, r, s', t, p) tuples experienced so far
        self.experience = np.empty((0, 6))

        # Path string to location for saving the memory data
        self.experience_path = experience_path

        # Load existing experience if override is false (and experience exists)
        if not override:
            if os.path.exists(experience_path):
                print("Reloading existing experience.")
                self.load_experience()
            else:
                print("No previous experience found at '%s'. A new experience dataset will be created." %
                      experience_path)
        else:
            if os.path.exists(experience_path):
                print("Overriding previously saved experience at %s." % experience_path)
                os.remove(experience_path)

        self.num_unsaved_transitions = 0

    def memorize(self, observations):
        self.experience = np.append(self.experience, observations, axis=0)

    def recall(self, num_transitions, alpha):
        """Returns a batch of transition IDs, depending on the transitions' priorities.
        This is part of Prioritized Experience Replay."""

        # Obtain number of experienced transitions
        exp_len = self.experience.shape[0]

        # Obtain batch size
        batch_size = np.min((exp_len, num_transitions))

        # Obtain priorities
        priorities = np.array(self.experience[:, -1], dtype='float64')

        # Take power of each element with alpha to adjust priorities
        priorities = np.power(priorities, alpha)

        # Convert priorities into probabilities
        probabilities = priorities / np.sum(priorities)

        # Randomly select transitions with given priorities (used as probabilities)
        trans_ids = np.random.choice(range(exp_len), size=batch_size, p=probabilities)

        return trans_ids, probabilities

    def get_length(self):
        return self.experience.shape[0]

    def export_all_experience(self, experience_path=None):
        """Exports the total experience data to a bzipped pickle file. For each level, the sequences of shots is saved.
                Each shot consists of the initial state, the chosen action and the achieved reward."""
        levels = []
        current_level = []

        if experience_path is None:
            experience_path = self.experience_path

        # Convert the experience into a more efficient format
        for state, action, reward, next_state, terminal, priority in self.experience:
            current_level += [state, action, reward, priority]

            if terminal:
                levels.append(current_level)
                current_level = []

        # Save it
        with bz2.open(experience_path, "wb") as f:
            for level in levels:
                pickle.dump(level, f)
        print("Stored", len(levels), "levels.")

    def export_new_experience(self):
        """Exports the experience data to a bzipped pickle file. For each level, the sequences of shots is saved.
        Each shot consists of the initial state, the chosen action and the achieved reward."""
        levels = []
        current_level = []
        num_shots = 0

        for i in range(self.num_unsaved_transitions):
            # i[0]: img, i[1]: action, i[2]: reward, i[3]: img, i[4]: termination, i[5]: termination
            current_offset = -self.num_unsaved_transitions + i
            current_level.append(self.experience[current_offset][0])
            current_level.append(self.experience[current_offset][1])
            current_level.append(self.experience[current_offset][2])
            current_level.append(self.experience[current_offset][5])
            num_shots += 1
            if self.experience[current_offset][4]:
                levels.append(current_level)
                current_level = []
                num_shots = 0

        # append the experience of the new levels to the experience file
        with bz2.open(self.experience_path, "ab") as f:
            _ = [pickle.dump(level, f) for level in levels]
        print("Stored", len(levels), "levels.")

        # reset the current episode_length
        self.num_unsaved_transitions = 0

    def load_experience(self, num_of_levels = -1):
        """Load the experience data from the compressed file and return it as a list of transitions.
        Each transition consists of: initial state, action reward, next state, termination.
        Loads the first num_of_levels from the file. If num_of_levels = -1, all levels are loaded."""

        if num_of_levels == -1:
            print("Try to load all levels.")
            num_of_levels = 200000
        else:
            print("Try to load", num_of_levels, "levels.")

        with bz2.open(self.experience_path, "rb") as f:
            levels = []
            for i in range(num_of_levels):
                if (i % 1000) == 0:
                    print("Loaded", i, "levels.")
                try:
                    levels.append(pickle.load(f))
                except EOFError:
                    break
            print("Finished loading. Loaded", len(levels), "levels in total.")

        experience = np.empty((0, 6))

        for l in levels:
            for i in range(int(len(l) / 4)):
                # add state, action, reward
                obs = [l[i * 4], l[i * 4 + 1], l[i * 4 + 2]]

                # check if the current transition is not the last transition
                if (i * 4 + 4) < len(l):
                    obs.append(l[i * 4 + 4])
                    obs.append(False)
                else:
                    obs.append(None)
                    obs.append(True)

                # append priority
                obs.append(l[i * 4 + 3])

                # add the current observation to the total list
                obs = np.array([obs])

                experience = np.append(experience, obs, axis=0)

        self.experience = experience

    def get_priorities(self):
        return self.experience[:, 5]

    def reset_priorities(self):
        self.experience[:, 5] = 1
