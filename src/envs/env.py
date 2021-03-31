import numpy as np


class Environment:
    """The basis class for all types of Reinforcement Learning environments."""
    # Environment name as string used for folder naming
    NAME = None

    # True if environment supports levels (e.g. Angry Birds does but Snake doesn't)
    LEVELS = None

    # Specification of relevant statistics (for console output during training and stat plots)
    TIME_RELEVANT = None
    WINS_RELEVANT = None


class ParallelEnvironment(Environment):
    """Class with the ability to simulate multiple environments in parallel."""
    def __init__(self, num_par_envs, actions):
        self.num_par_envs = num_par_envs
        self.actions = actions  # List of action names (strings)

        self.game_overs = np.zeros(shape=num_par_envs, dtype="bool")
        self.rewards = np.zeros(shape=num_par_envs, dtype="float32")
        self.scores = np.zeros(shape=num_par_envs, dtype="int32")
        self.times = np.zeros(shape=num_par_envs, dtype="uint16")
        self.wins = np.zeros(shape=num_par_envs, dtype="bool")  # for envs with levels

    def reset(self):
        """Resets all environments to their initial state."""
        pass

    def reset_for(self, ids):
        """Resets selected environments to their initial state."""
        pass

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
        pass

    def get_states(self):
        """Returns a state representation for all environments in a single NumPy array."""
        pass

    def get_state_shapes(self):
        """Returns two arguments:
        image_state_shape: dimensions of the image state matrix in channel-last order
        numerical_state_shape: length of the numerical state vector"""
        pass

    def get_number_of_actions(self):
        """Returns the number of possible actions performable in the environment. There
        are environments where the number of actions is determined only after initialization."""
        return len(self.actions)

    def render(self):
        """Renders the environment inside a PyGame window."""
        pass

    def image_state_to_text(self, image_state):
        """Returns a simple textual visualization of a given image state."""
        return ""

    def numerical_state_to_text(self, numerical_state):
        """Returns a simple textual visualization of a given numerical state."""
        return ""

    def set_mode(self, mode):
        """Sets the level selection/generation mode."""
        print("This environment doesn't support different modes!")

    def generate_pretrain_data(self, num_instances):
        """Generates a set of num_instances images which can be used for autoencoder pretraining."""
        pass

    def has_test_levels(self):
        """Returns True if the env has dedicated test levels."""
        return False


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

    def image_state_to_text(self, image_state):
        """Returns a simple textual visualization of a given image state."""
        return ""

    def numerical_state_to_text(self, numerical_state):
        """Returns a simple textual visualization of a given numerical state."""
        return ""
