class Environment:
    """The basis class for all types of Reinforcement Learning environments."""
    name = None


class ParallelEnvironment(Environment):
    """Class with the ability to simulate multiple environments in parallel."""
    def __init__(self, num_par_envs, actions):
        self.num_par_envs = num_par_envs
        self.actions = actions  # List of action names (strings)

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
        """
        pass

    def get_states(self):
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
