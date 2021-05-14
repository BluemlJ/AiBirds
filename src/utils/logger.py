from src.utils.text_sty import num2text
from tabulate import tabulate
import os
import numpy as np


class Logger:
    """Manages log files: creates, writes and closes them."""

    def __init__(self, out_path, extreme_loss_factor, window_size_cycles):
        self.step_stats_file = open(out_path + "step_stats.txt", "a", buffering=1)
        self.records_file = open(out_path + "records.txt", "a", buffering=1)
        self.extreme_loss_file = open(out_path + "extreme_losses.txt", "a", buffering=1)
        if os.stat(out_path + "extreme_losses.txt").st_size == 0:
            self.extreme_loss_file.write("Here, training instances with a loss larger than %.0f times the"
                                         "\n%s MA loss of the most recent training cycles will be logged.\n"
                                         % (extreme_loss_factor, num2text(window_size_cycles)))
        self.log_file = open(out_path + "log.txt", "a", buffering=1)
        self.window_size_cycles = window_size_cycles

    def log_step_statistics(self, step_statistics):
        self.step_stats_file.write(step_statistics)

    def log_new_record(self, obs_return, transition):
        text = "New return record achieved! The new best return is %.2f.\n" % obs_return + \
               "This is how the episode ended:" + transition + "\n\n"
        self.records_file.write(text)

    def log_extreme_loss(self, train_cycle, loss, ma_loss, pred_q, target_q, rewards, mask,
                         trans_text, actions):
        text = "\nExtreme loss encountered in train cycle %d." % train_cycle
        text += "\nLoss: %.4f, MA loss: %.4f" % (loss, ma_loss)
        text += "\n" + tabulate([["Predicted", *pred_q], ["Target", *target_q]],
                                headers=["Q-values", *actions], floatfmt=".2f", tablefmt="rst")
        text += "\n" + tabulate([["Rewards", *rewards], ["Mask", *mask]],
                                headers=["n-step", *np.arange(1, len(rewards) + 1)],
                                floatfmt=".2f", tablefmt="rst")
        text += trans_text
        self.extreme_loss_file.write(text)

    def log(self, text):
        self.log_file.write(text)

    def close(self):
        self.step_stats_file.close()
        self.records_file.close()
        self.extreme_loss_file.close()
        self.log_file.close()
