class DecayParam:
    """Features linear warmup and exponential or linear decay."""

    def __init__(self, init_value, decay_mode=None, half_life_period=None, warmup_transitions=0,
                 minimum=0):
        """
        :param init_value: Initial value of epsilon
        :param decay_mode: The function used to decrease epsilon after each parallel step ("lin" or "exp")
        :param half_life_period: Number of transitions after which value reaches half of its initial value.
        :param warmup_transitions: Number of transitions for linear warmup. May be useful for learning rate.
        :param minimum: Minimum value which is never undercut over the course of practice
        """
        assert init_value >= minimum
        self.init_value = init_value
        self.warmup_transitions = warmup_transitions
        self.half_life_period = half_life_period
        self.decaying = decay_mode is not None
        self.decay_mode = decay_mode
        self.decay_rate = 0.5 ** (1 / half_life_period) if self.decaying else None
        self.minimum = minimum

    def get_value(self, current_trans_no):
        if current_trans_no < self.warmup_transitions:
            return self.init_value * current_trans_no / self.warmup_transitions
        elif self.decaying:
            decay_transitions = current_trans_no - self.warmup_transitions
            if self.decay_mode == "exp":
                return max(self.init_value * self.decay_rate ** decay_transitions, self.minimum)
            else:
                return max(self.init_value * (1 - decay_transitions / (2 * self.half_life_period)), self.minimum)
        else:
            return self.init_value

    def get_config(self):
        config = {"init_value": self.init_value,
                  "half_life_period": self.half_life_period,
                  "self.decay_mode": self.decay_mode,
                  "warmup_transitions": self.warmup_transitions,
                  "minimum": self.minimum}
        return config
