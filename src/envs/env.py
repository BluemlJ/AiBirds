import numpy as np
from abc import ABCMeta
from typing import List


class Environment(metaclass=ABCMeta):
    """The basis class for all types of Reinforcement Learning environments."""
    # Environment name as string used for folder naming
    NAME = None

    # True if environment supports levels (e.g. Angry Birds does but Snake doesn't)
    LEVELS = None

    # Specification of relevant statistics (for console output during training and stat plots)
    TIME_RELEVANT = None
    WINS_RELEVANT = None

    def get_config(self):
        pass


class ParallelEnvironment(Environment, metaclass=ABCMeta):
    """Class with the ability to simulate multiple environments in parallel."""
    def __init__(self, num_par_inst, actions):
        self.num_par_inst = num_par_inst
        self.actions = actions  # List of action names (strings)

        self.scores = np.zeros(shape=num_par_inst, dtype="int32")
        self.times = np.zeros(shape=num_par_inst, dtype="uint16")
        self.game_overs = np.zeros(shape=num_par_inst, dtype="bool")
        self.wins = np.zeros(shape=num_par_inst, dtype="bool")  # for envs with levels

    def reset(self):
        """Resets all environments to their initial state."""
        self.scores[:] = 0
        self.times[:] = 0
        self.game_overs[:] = False
        self.wins[:] = False

    def reset_for(self, ids):
        """Resets selected environments to their initial state."""
        self.scores[ids] = 0
        self.times[ids] = 0
        self.game_overs[ids] = False
        self.wins[ids] = False

    def step(self, actions):
        """The given actions are executed in the environment and the environment
        performs a time step.

        :return: Updated information about all environments:
            rewards: the rewards gained with this step
            scores: the current raw game scores of all environments
            game_overs: a Boolean array indicating game overs
            times: an int array indicating for each env the number of steps since the last reset
            wins: a Boolean array indicating for each env if the episode was won (if defined)
        """
        raise NotImplementedError

    def get_states(self) -> List[np.array]:
        """Returns state representations for all parallel environments. Each element of the
        returned list is a NumPy array, containing the same-shape state components of all
        parallel env instances."""
        raise NotImplementedError

    def get_state_shapes(self):
        """Returns two arguments:
        image_state_shape: dimensions of the image state matrix in channel-last order
        numerical_state_shape: length of the numerical state vector"""
        raise NotImplementedError

    def get_number_of_actions(self):
        """Returns the number of possible actions performable in the environment. There
        are environments where the number of actions is determined only after initialization."""
        return len(self.actions)

    def render(self):
        """Renders the environment inside a PyGame window."""
        raise NotImplementedError

    def state2text(self, state):
        """Returns a simple textual visualization of a given state."""
        return "State representation not implemented."

    def print_all_current_states(self):
        states = self.get_states()
        for env_id in self.num_par_inst:
            state = [state_comps[env_id] for state_comps in states]
            print("Environment instance %d:" % env_id)
            print(self.state2text(state))

    def set_mode(self, mode):
        """Sets the level selection/generation mode."""
        print("This environment doesn't support different modes!")

    def generate_pretrain_data(self, num_instances):
        """Generates a set of num_instances images which can be used for autoencoder pretraining."""
        raise NotImplementedError

    def has_test_levels(self):
        """Returns True if the env has dedicated test levels."""
        return False

    def get_config(self):
        return {"num_par_inst": self.num_par_inst}

    def copy(self, num_par_inst):
        config = self.get_config()
        config.pop("num_par_inst")
        env_type = type(self)
        return env_type(num_par_inst=num_par_inst, **config)


class MultiAgentEnvironment(Environment):
    """An abstract class for environments with multiple agents."""

    def __init__(self, num_agents, actions):
        self.num_agents = num_agents
        self.actions = actions  # List of action names (strings)

    def reset(self):
        """Resets the env to its initial state."""
        pass

    def step(self, actions):
        """The given actions are executed in the environment for each agent in parallel and the environment
        performs a time step.

        :return: Updated information about all environments:
            rewards: the rewards gained with this step
            scores: the current raw game scores of all environments
            game_overs: a Boolean array indicating game overs
            times: an int array indicating for each env the number of steps since the last reset
        """
        pass

    def get_state(self):
        """Returns a state representation for all environments in a single NumPy array."""
        pass

    def get_state_shape(self):
        """Returns two arguments:
        image_state_shape: dimensions of the image state matrix in channel-last order
        numerical_state_shape: length of the numerical state vector"""
        pass

    def get_number_of_actions(self):
        """Returns the number of possible actions performable in the environment."""
        return len(self.actions)

    def render(self):
        """Renders the environment inside a PyGame window."""
        pass

    def state2text(self, state):
        """Returns a simple, human-readable, textual visualization of a given state."""
        return ""
