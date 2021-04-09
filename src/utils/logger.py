from src.utils.utils import num2text
import os


class Logger:
    """Manages log files: creates, writes and closes them."""

    def __init__(self, out_path, extreme_loss_factor, window_size_cycles):
        self.step_stats_file = open(out_path + "step_stats.txt", "a", buffering=1)
        self.records_file = open(out_path + "records.txt", "a", buffering=1)
        self.extreme_loss_file = open(out_path + "extreme_losses.txt", "a", buffering=1)
        if os.stat(out_path + "extreme_losses.txt").st_size == 0:
            self.extreme_loss_file.write("Here, training instances with a loss larger than %.0f times the "
                                         "%s MA loss of the most recent training cycles will be logged.\n\n"
                                         % (extreme_loss_factor, num2text(window_size_cycles)))
        self.log_file = open(out_path + "log.txt", "a", buffering=1)
        self.window_size_cycles = window_size_cycles

    def log_step_statistics(self, step_statistics):
        self.step_stats_file.write(step_statistics)

    def log_new_record(self, obs_return, transition):
        text = "New return record achieved! The new best return is %.2f.\n" % obs_return + \
               "This is how the episode ended:" + transition + "\n\n"
        self.records_file.write(text)

    def log_extreme_loss(self, train_cycle, loss, ma_loss, pred_q, target_q, trans_text):
        text = "Extreme loss encountered in train cycle %d." % train_cycle + \
               "\nExample loss: %.4f" % loss + \
               "\n%s MA loss: %.4f" % (num2text(self.window_size_cycles), ma_loss) + \
               "\nPredicted Q-values: " + pred_q + \
               "\nTarget Q-values: " + target_q + \
               trans_text + "\n"
        self.extreme_loss_file.write(text)

    def log(self, text):
        self.log_file.write(text)

    def close(self):
        self.step_stats_file.close()
        self.records_file.close()
        self.extreme_loss_file.close()
        self.log_file.close()
