import numpy as np


class ParamScheduler:
    """Features linear warmup, exponential or linear decay and step milestones."""

    def __init__(self, init_value, decay_mode=None, half_life_period=None, warmup_transitions=0,
                 minimum=0, milestones=None, milestone_factor=None):
        """
        :param init_value: Initial value of epsilon
        :param decay_mode: The function used to decrease epsilon after each parallel step ("lin" or "exp")
        :param half_life_period: Number of transitions after which value reaches half of its initial value.
        :param warmup_transitions: Number of transitions for linear warmup. May be useful for learning rate.
        :param minimum: Minimum value which is never undercut over the course of practice
        :param milestones: Used in step decay. The steps after each the current value is multiplied by
               milestone_factor.
        :param milestone_factor: Used in step decay. The factor applied after each milestone.
        """
        assert init_value >= minimum
        self.init_value = init_value
        self.warmup_transitions = warmup_transitions
        self.half_life_period = half_life_period
        if decay_mode in ["exp", "lin"]:
            assert half_life_period is not None
            self.decay_rate = 0.5 ** (1 / half_life_period)
        elif decay_mode == "step":
            assert milestones is not None and milestone_factor is not None
            assert 0 < milestone_factor < 1
            self.decay_rate = None
        elif decay_mode is not None:
            raise ValueError("Invalid decay mode given:", decay_mode)
        self.decaying = decay_mode is not None
        self.decay_mode = decay_mode
        self.minimum = minimum
        self.milestones = np.array(milestones)
        self.milestone_factor = milestone_factor

    def get_value(self, current_trans_no):
        if current_trans_no < self.warmup_transitions:
            return self.init_value * current_trans_no / self.warmup_transitions
        elif self.decaying:
            decay_transitions = current_trans_no - self.warmup_transitions
            if self.decay_mode == "exp":
                return max(self.init_value * self.decay_rate ** decay_transitions, self.minimum)
            elif self.decay_mode == "lin":
                return max(self.init_value * (1 - decay_transitions / (2 * self.half_life_period)), self.minimum)
            elif self.decay_mode == "step":
                milestone_cnt = np.sum(self.milestones <= current_trans_no)
                return self.init_value * self.milestone_factor ** milestone_cnt
        else:
            return self.init_value

    def get_config(self):
        config = {"init_value": self.init_value,
                  "self.decay_mode": self.decay_mode}
        if self.warmup_transitions > 0:
            config.update({"warmup_transitions": self.warmup_transitions})
        if self.decaying:
            if self.decay_mode in ["exp", "lin"]:
                config.update({"half_life_period": self.half_life_period,
                               "minimum": self.minimum})
            elif self.decay_mode == "step":
                config.update({"milestones": self.milestones,
                               "milestone_factor": self.milestone_factor})
        return config
