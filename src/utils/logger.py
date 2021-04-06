from src.utils.stats import EXTREME_LOSS_FACTOR, WINDOW_SIZES_LOSS


class Logger:
    """Manages log files: creates, writes and closes them."""

    def __init__(self, out_path):
        self.step_stats_file = open(out_path + "step_stats.txt", "a", buffering=1)
        self.records_file = open(out_path + "records.txt", "a", buffering=1)
        self.extreme_loss_file = open(out_path + "extreme_losses.txt", "a", buffering=1)
        self.extreme_loss_file.write("Here, training instances with a loss larger than %.0f times the "
                                     "%d-moving average loss of the most recent training cycles will be logged.\n\n"
                                     % (EXTREME_LOSS_FACTOR, WINDOW_SIZES_LOSS[0]))
        self.log_file = open(out_path + "log.txt", "a", buffering=1)

    def log_step_statistics(self, step_statistics):
        self.step_stats_file.write(step_statistics)

    def log_new_record(self, obs_return, transition):
        text = "New return record achieved! The new best return is %.2f.\n" % obs_return + \
               "This is how the episode ended:" + transition + "\n\n"
        self.records_file.write(text)

    def log_extreme_loss(self, train_cycle, loss, ma_loss, pred_q, target_q, trans_text):
        text = "Extreme loss encountered in train cycle %d." % train_cycle + \
               "\nExample loss: %.4f" % loss + \
               "\n%d moving avg. loss: %.4f" % (WINDOW_SIZES_LOSS[0], ma_loss) + \
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
