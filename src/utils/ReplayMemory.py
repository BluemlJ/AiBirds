import numpy as np
import pickle
import bz2
import os


class ReplayMemory:
    def __init__(self, minibatch=32, experience_path="data/experiences.bz2", override=False):
        # Initialize the experience list, a list of transitions
        # containing all (s, a, r, s', t) tuples experienced so far
        self.experience = np.empty((0, 6))

        # Path string to location for saving the memory data
        self.experience_path = experience_path

        # Load existing experience if override is false (and experience exists)
        if not override:
            print("Reloading existing experience.")
            self.load_experience()
        else:
            print("Overriding previously saved experience.")
            if os.path.exists(experience_path):
                os.remove(experience_path)

        self.current_episode_length = 0

        # Number of transitions to return when recall is called
        self.minibatch = minibatch

    def memorize(self, observations):
        self.experience = np.append(self.experience, observations, axis=0)

    def recall(self):
        """Returns a batch of useful transitions. This uses Prioritized Experience Replay."""

        # Obtain number of experienced transitions
        exp_len = self.experience.shape[0]

        # Obtain batch size
        batch_size = np.min((exp_len, self.minibatch))

        for j in range(self.minibatch):
            # TODO: Sample transition depending on priorities
            break

        # Select a random batch from experience to train on, TODO: implement Expereince Replay
        batch_ids = np.random.choice(exp_len, batch_size)
        batch = self.experience[batch_ids]

        return batch

    def export_all_experience(self):
        """Exports the total experience data to a bzipped pickle file. For each level, the sequences of shots is saved.
                Each shot consists of the initial state, the chosen action and the achieved reward."""
        levels = []
        current_level = []

        # Convert the experience into a more efficient format
        for state, action, reward, next_state, terminal, priority in self.experience:
            current_level += [state, action, reward, priority]

            if terminal:
                levels.append(current_level)
                current_level = []

        # Save it
        with bz2.open(self.experience_path, "wb") as f:
            pickle.dump(levels, f)
        print("Stored", len(levels), "levels.")

    def export_new_experience(self):
        """Exports the experience data to a bzipped pickle file. For each level, the sequences of shots is saved.
        Each shot consists of the initial state, the chosen action and the achieved reward."""
        levels = []
        current_level = []
        num_shots = 0

        for i in range(self.current_episode_length):
            # i[0]: img, i[1]: action, i[2]: reward, i[3]: img, i[4]: termination, i[5]: termination
            current_offset = -self.current_episode_length + i
            current_level.append(self.experience[current_offset][0])
            current_level.append(self.experience[current_offset][1])
            current_level.append(self.experience[current_offset][2])
            current_level.append(self.experience[current_offset][5])
            num_shots += 1
            if self.experience[current_offset][4]:
                levels.append(current_level)
                current_level = []
                num_shots = 0

        # try to open the file with previous experiences, if not possible: create an empty experience list
        try:
            with bz2.open(self.experience_path, "rb") as f:
                pre_levels = pickle.load(f)
                print("Loaded:", len(pre_levels), "levels.")
        except EOFError:
            pre_levels = []
        except FileNotFoundError:
            pre_levels = []

        pre_levels.extend(levels)

        with bz2.open(self.experience_path, "wb") as f:
            pickle.dump(pre_levels, f)
        print("Stored", len(pre_levels), "levels.")

        # reset the current episode_length
        self.current_episode_length = 0

    def load_experience(self):
        """Load the experience data from the compressed file and return it as a list of transitions.
        Each transition consists of: initial state, action reward, next state, termination."""
        with bz2.open(self.experience_path, "rb") as f:
            levels = pickle.load(f)
            print("Loaded", len(levels), "levels.")

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
