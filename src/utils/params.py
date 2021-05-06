import numpy as np


class ParamScheduler:
    """Features linear warmup, exponential or linear decay and step milestones."""

    def __init__(self, init_value, decay_mode=None, half_life_period=None, warmup_transitions=0,
                 minimum=0, milestones=None, milestone_factor=None, milestone_values=None):
        """
        :param init_value: Initial value of epsilon
        :param decay_mode: The function used to decrease epsilon after each parallel step ("lin" or "exp")
        :param half_life_period: Number of transitions after which value reaches half of its initial value.
        :param warmup_transitions: Number of transitions for linear warmup. May be useful for learning rate.
        :param minimum: Minimum value which is never undercut over the course of practice
        :param milestones: Used in step decay. The steps after each the current value is multiplied by
               milestone_factor.
        :param milestone_factor: Used in step decay. The factor applied after each milestone.
        :param milestone_values: Values at the corresponding milestones. List of same length as milestones.
        """
        # Check user inputs
        assert init_value >= minimum, "Initial value must lie above minimum value."

        if decay_mode == "lin":
            assert milestones is not None, "Linear decay mode requires milestones."
            assert milestone_values is not None, "Linear decay mode requires milestone values."
        elif decay_mode == "exp":
            assert half_life_period is not None
        elif decay_mode == "step":
            assert milestones is not None and (milestone_factor is not None or milestone_values is not None)
            assert milestone_factor is None or 0 < milestone_factor < 1
        elif decay_mode is not None:
            raise ValueError("Invalid decay mode given:", decay_mode)

        self.init_value = init_value
        self.dynamic = decay_mode is not None
        self.decay_mode = decay_mode

        assert warmup_transitions is None or milestones is None or milestones[0] > warmup_transitions
        self.warmup_transitions = warmup_transitions

        self.half_life_period = half_life_period
        self.decay_rate = 0.5 ** (1 / half_life_period) if half_life_period is not None else None

        assert milestone_factor is None or milestone_values is None, \
            "Decay accepts either factor or step values, not both."
        assert milestone_values is None or len(milestone_values) == len(milestones)
        self.milestones = np.array(milestones) if milestones is not None else None
        self.milestone_factor = milestone_factor
        self.milestone_values = np.array(milestone_values) if milestone_values is not None else None

        assert self.milestone_values is None or np.all(self.milestone_values >= minimum), \
            "Milestone values are required to lie above the global minimum."
        self.minimum = minimum

    def get_value(self, current_trans_no):
        if current_trans_no < self.warmup_transitions:
            return self.init_value * current_trans_no / self.warmup_transitions
        elif self.dynamic:
            decay_transitions = current_trans_no - self.warmup_transitions
            if self.decay_mode == "exp":
                return max(self.init_value * self.decay_rate ** decay_transitions, self.minimum)
            else:
                milestone_cnt = np.sum(self.milestones <= current_trans_no)
                if milestone_cnt == len(self.milestones) and self.milestone_values is not None:
                    return self.milestone_values[-1]
                elif self.decay_mode == "lin":
                    # Linear interpolation
                    if milestone_cnt == 0:
                        segment_start = 0
                        start_value = self.init_value
                    else:
                        segment_start = self.milestones[milestone_cnt - 1]
                        start_value = self.milestone_values[milestone_cnt - 1]

                    end_value = self.milestone_values[milestone_cnt]
                    segment_end = self.milestones[milestone_cnt]

                    segment_progress = (current_trans_no - segment_start) / (segment_end - segment_start)
                    return start_value * (1 - segment_progress) + end_value * segment_progress
                elif self.decay_mode == "step":
                    if milestone_cnt == 0:
                        return self.init_value
                    elif self.milestone_factor is not None:
                        return self.init_value * self.milestone_factor ** milestone_cnt
                    else:
                        return self.milestone_values[milestone_cnt - 1]
        else:
            return self.init_value

    def get_config(self):
        config = {"init_value": self.init_value,
                  "decay_mode": self.decay_mode}
        if self.warmup_transitions > 0:
            config.update({"warmup_transitions": self.warmup_transitions})
        if self.dynamic:
            if self.decay_mode == "exp":
                config.update({"half_life_period": self.half_life_period,
                               "minimum": self.minimum})
            elif self.decay_mode in ["step", "lin"]:
                config.update({"milestones": self.milestones,
                               "milestone_values": self.milestone_values,
                               "milestone_factor": self.milestone_factor})
        return config
