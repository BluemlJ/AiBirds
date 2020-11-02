class ParallelEnvironment:
    """A parent class for any Reinforcement Learning game environment, with the
    ability to simulate multiple environments in parallel."""

    def __init__(self, name, num_par_envs, actions):
        self.name = name
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
            states: a NumPy array containing current state of all environments
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

    def get_name(self):
        return self.name
