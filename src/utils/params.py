import numpy as np


class Epsilon:
    """A simple class providing basic functions to handle decaying epsilon greedy exploration."""

    def __init__(self, init_value, decay_mode="exp", decay_rate=1, minimum=0):
        """
        :param init_value: Initial value of epsilon
        :param decay_mode: The function used to decrease epsilon after each parallel step ("lin" or "exp")
        :param decay_rate: Decrease/anneal factor for epsilon used when decay() gets invoked
        :param minimum: Minimum value for epsilon which is never undercut over thr course of practice
        """
        if init_value < minimum:
            raise ValueError("You must provide a value for epsilon larger than the minimum.")

        if decay_mode not in ["exp", "lin"]:
            raise ValueError("Invalid decay mode provided. You gave %s, but only 'exp' and 'lin' are allowed." %
                             decay_mode)

        self.init_value = init_value
        self.decay_mode = decay_mode
        self.volatile_val = init_value - minimum
        self.rigid_val = minimum
        self.decay_rate = decay_rate
        if decay_mode == "exp":
            self.decay_fn = self.decay_exp
        else:
            self.decay_fn = self.decay_lin
        self.minimum = minimum

    def get_value(self):
        return self.volatile_val + self.rigid_val

    def get_decay(self):
        return self.decay_rate

    def get_minimum(self):
        return self.minimum

    def decay(self):
        self.volatile_val = self.decay_fn(self.volatile_val)

    def decay_exp(self, val):
        return self.decay_rate * val

    def decay_lin(self, val):
        return np.max([val - self.decay_rate, self.rigid_val])

    def set_value(self, value, minimum=None):
        if minimum is None:
            minimum = self.minimum
        self.volatile_val = value - minimum
        self.rigid_val = minimum

    def get_config(self):
        config = {"init_value": self.init_value,
                  "decay_mode": self.decay_mode,
                  "decay_rate": self.decay_rate,
                  "minimum": self.minimum}
        return config


class LearningRate:
    """Custom learning rate scheduler, depending on number of current episode (in
    contrast to current optimizer step). Features linear warmup and exponential decay."""

    def __init__(self, initial_learning_rate, warmup_episodes=0, half_life_period=None):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_episodes = warmup_episodes
        self.half_life_period = half_life_period
        self.decay_rate = None if half_life_period is None else 0.5 ** (1 / half_life_period)

    def get_value(self, current_episode_no):
        if current_episode_no < 0:
            raise ValueError("Invalid episode number given. Number must be non-negative int.")

        if current_episode_no < self.warmup_episodes:
            return self.initial_learning_rate * current_episode_no / self.warmup_episodes
        elif self.half_life_period is not None:
            decay_episodes = current_episode_no - self.warmup_episodes
            return self.initial_learning_rate * self.decay_rate ** decay_episodes
        else:
            return self.initial_learning_rate

    def get_config(self):
        config = {"initial_learning_rate": self.initial_learning_rate,
                  "warmup_episodes": self.warmup_episodes,
                  "half_life_period": self.half_life_period}
        return config
