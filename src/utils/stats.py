import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from src.utils.utils import orange, yellow, bold, plot, user_agrees_to, convert_secs_to_hhmmss, num2text
from src.envs.env import Environment

WINDOW_SIZES_EPISODES = (1000, 10000, 50000)
WINDOW_SIZES_LOSS = (10, 50, 200)
EXTREME_LOSS_FACTOR = 500


class Statistics:
    def __init__(self, env: Environment = None):
        self.final_returns = []
        self.final_scores = []
        self.final_times = []
        self.train_losses = []
        self.learning_rates = []
        self.wins = []

        self.current_run_episode_no = 0

        self.return_records = []
        self.score_records = []
        self.time_records = []

        self.ma_score_record = 0

        if env is not None:
            self.time_relevant = env.TIME_RELEVANT
            self.win_relevant = env.WINS_RELEVANT
        else:
            self.time_relevant = False
            self.win_relevant = False

        self.continue_training = False

    def denote_stats(self, final_return, final_score, final_time, win):
        self.final_returns += [final_return]
        self.final_scores += [final_score]
        if self.time_relevant:
            self.final_times += [final_time]
        if self.win_relevant:
            self.wins += [win]

        return_record, score_record, time_record = self.get_records()
        self.return_records += [np.max((return_record, final_return))]
        self.score_records += [np.max((score_record, final_score))]
        self.time_records += [np.max((time_record, final_time))]

        self.current_run_episode_no += 1

        return return_record < final_return

    def denote_loss(self, loss):
        self.train_losses += [loss]

    def denote_learning_rate(self, learning_rate):
        self.learning_rates += [learning_rate]

    def denote_learning_stats(self, loss, individual_losses, learning_rate, trans_ids, predictions,
                              targets, memory, env, logger):
        self.denote_loss(loss)
        self.denote_learning_rate(learning_rate)
        self.log_extreme_losses(individual_losses, trans_ids, predictions, targets,
                                      memory, env, logger)

    def get_length(self):
        return len(self.final_scores)

    def get_train_cycle(self):
        return len(self.train_losses)

    def get_final_returns(self):
        return np.array(self.final_returns)

    def get_final_scores(self):
        return np.array(self.final_scores)

    def get_final_times(self):
        return np.array(self.final_times)

    def get_wins(self):
        if self.win_relevant:
            return np.array(self.wins)
        else:
            return []

    def get_train_losses(self):
        return np.array(self.train_losses)

    def get_return_records(self):
        return np.array(self.return_records)

    def get_score_records(self):
        return np.array(self.score_records)

    def get_time_records(self):
        if self.time_relevant:
            return np.array(self.time_records)
        else:
            return []

    def get_current_score(self):
        if len(self.final_scores) > 0:
            return self.final_scores[-1]
        else:
            return None

    def get_learning_rates(self):
        return np.array(self.learning_rates)

    def get_current_learning_rate(self):
        if len(self.learning_rates) == 0:
            return np.nan
        else:
            return self.learning_rates[-1]

    def get_records(self):
        if len(self.return_records) == 0:
            return -np.inf, 0, 0
        else:
            if self.time_relevant:
                time_record = self.time_records[-1]
            else:
                time_record = 0
            return self.return_records[-1], self.score_records[-1], time_record

    def compute_records(self):
        num_episodes = self.get_length()
        self.return_records = []
        self.score_records = []
        if self.time_relevant:
            self.time_records = []

        for i in range(num_episodes):
            return_record, score_record, time_record = self.get_records()
            self.return_records += [np.max((return_record, self.final_returns[i]), initial=-np.inf)]
            self.score_records += [np.max((score_record, self.final_scores[i]), initial=0)]
            if self.time_relevant:
                self.time_records += [np.max((time_record, self.final_times[i]), initial=0)]

    def get_stats_dict(self):
        return {"returns": self.final_returns, "scores": self.final_scores,
                "times": self.final_times, "losses": self.train_losses,
                "wins": self.wins, "learning_rates": self.learning_rates}

    def set_stats(self, stats_dict):
        self.final_returns = stats_dict["returns"]
        self.final_scores = stats_dict["scores"]
        if self.time_relevant:
            self.final_times = stats_dict["times"]
        self.train_losses = stats_dict["losses"]
        if self.win_relevant:
            self.wins = stats_dict["wins"]
        self.compute_records()
        try:
            self.learning_rates = stats_dict["learning_rates"]
        except Exception as e:
            print("Learning rate values missing or corrupted.")
            print(e)

    def save(self, out_path):
        file_path = out_path + "stats.pckl"
        stats_dict = self.get_stats_dict()
        with open(file_path, 'wb') as json_file:
            pickle.dump(stats_dict, json_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, in_path):
        file_path = in_path + "stats.pckl"
        with open(file_path, 'rb') as json_file:
            try:
                stats_dict = pickle.load(json_file)
                self.set_stats(stats_dict)
            except Exception as e:
                print("Couldn't load statistics from '%s'!" % in_path)
                print(e)
            else:
                print("Successfully loaded statistics from '%s'." % in_path)

    def progress_stagnates(self):
        """Returns true if loss did not change much and score did not increase."""
        return self.loss_stagnates() and self.score_stagnates()

    def loss_stagnates(self):
        if len(self.train_losses) >= 1000:
            magnitude = np.average(self.train_losses[-1000:])
            long_term_stagnation = np.abs(np.average(self.train_losses[-1000:-500]) -
                                          np.average(self.train_losses[-500:])) < 0.005 * magnitude
            short_term_stagnation = np.abs(np.average(self.train_losses[-400:-200]) -
                                           np.average(self.train_losses[-200:])) < 0.005 * magnitude
            return long_term_stagnation and short_term_stagnation
        else:
            return False

    def score_stagnates(self):
        """Returns true if score improvement was low. More precisely, returns True if the 1000 MA score did improve
        by less than 5% during the last 4000 episodes."""
        scores = self.get_final_scores()
        if len(scores) >= 5000:
            avg_score_old = np.average(scores[-5000:-4000])
            avg_score_new = np.average(scores[-1000:])
            return avg_score_new / avg_score_old < 1.05
        else:
            return False

    def score_crashed(self, ma_score):
        if ma_score is not None:
            if ma_score < 0.05 * self.ma_score_record:
                return True
        return False

    def print_stats(self, par_step, total_par_steps, num_envs, comp_time, total_comp_time, print_stats_period,
                    epsilon, logger, ma_window_size=1000):
        # avg_return = get_moving_avg(self.get_final_returns(), ma_window_size)
        ma_score = get_moving_avg_val(self.get_final_scores(), ma_window_size)
        ma_loss = get_moving_avg_val(self.get_train_losses(), 50)
        self.ma_score_record = np.max((self.ma_score_record, ma_score))

        return_record, score_record, time_record = self.get_records()
        done_transitions = par_step * num_envs
        ma_text = num2text(ma_window_size)

        stats_text = "\n" + bold("Parallel step %d/%d" % (par_step, total_par_steps)) + \
                     "\n   Completed episodes:        %s" % num2text(self.get_length()) + \
                     "\n   Transitions done:          %s" % num2text(done_transitions) + \
                     "\n   Epsilon:                   %.3f" % epsilon.get_value() + \
                     "\n   " + "{:27s}".format("Score (%s MA):" % ma_text) + ("%.1f" % ma_score) + \
                     "\n   Score record:              %.0f" % score_record

        if self.win_relevant:
            avg_wins = get_moving_avg_val(self.get_wins(), ma_window_size)
            stats_text += \
                "\n   " + "{:27s}".format("Win-ratio (%s MA):" % ma_text) + ("%.2f" % avg_wins)

        if self.time_relevant:
            avg_time = get_moving_avg_val(self.get_final_times(), ma_window_size)
            stats_text += \
                "\n   " + "{:27s}".format("Time (%s MA):" % ma_text) + ("%.1f" % avg_time) + \
                "\n   Time record:               %.0f" % time_record

        stats_text += "\n   Loss (50 MA):              %.4f" % ma_loss + \
                      "\n   Learning rate:             %.6f" % self.get_current_learning_rate() + \
                      "\n------" \
                      "\n   " + "{:27s}".format("Comp time (last %d):" % print_stats_period) + \
                      "%d s" % np.round(comp_time) + \
                      "\n   Total comp time:           " + convert_secs_to_hhmmss(total_comp_time)

        print(stats_text)
        logger.log_step_statistics(stats_text + "\n")

        if self.score_crashed(ma_score) and not self.continue_training:
            question = orange("Score crashed! %s MA score dropped by 95 %%. "
                              "Still want to continue training? (y/n)" % ma_text)
            if not user_agrees_to(question):
                quit()
            else:
                self.continue_training = True

    def plot_stats(self, out_path, memory):
        if self.get_length() >= 200:
            # Plot priorities bar chart
            plt.hist(memory.get_priorities(), range=None, bins=100)
            plot(title="Transition priorities in experience set",
                 xlabel="Priority value",
                 ylabel="Number of transitions",
                 output_path=out_path + "priorities.png")

            # Plot train loss line chart
            if self.get_train_cycle() > WINDOW_SIZES_LOSS[0]:
                plot_moving_average(self.get_train_losses(),
                                    title="Training loss history",
                                    window_sizes=WINDOW_SIZES_LOSS,
                                    ylabel="Loss", xlabel="Train cycle",
                                    output_path=out_path + "loss.png",
                                    logarithmic=True)

            # Plot learning rate line chart
            if self.get_current_learning_rate() is not np.nan:
                plt.plot(self.get_learning_rates())
                plt.ylim(bottom=0)
                plot(title="Learning rate history",
                     xlabel="Train cycle", ylabel="Learning rate",
                     output_path=out_path + "learning_rates.png",
                     legend=False,
                     logarithmic=False)

            # Plot time line chart
            if self.time_relevant:
                plot_moving_average(self.get_final_times(),
                                    title="Episode length history",
                                    ylabel="Time (game ticks)",
                                    output_path=out_path + "times.png")
                plt.plot(self.time_records)
                plot(title="Episode length records",
                     xlabel="Episode",
                     ylabel="Time (game ticks)",
                     output_path=out_path + "time_records.png")

            # Plot score line chart
            final_scores = self.get_final_scores()
            if len(final_scores) > 0:
                plot_moving_average(final_scores,
                                    title="Score history",
                                    ylabel="Score",
                                    output_path=out_path + "scores.png",
                                    show=True)
                plt.plot(self.score_records)
                plot(title="Score records",
                     xlabel="Episode",
                     ylabel="Score",
                     output_path=out_path + "score_records.png")

            # Plot return line chart
            final_returns = self.get_final_returns()
            if len(final_returns) > 0:
                plot_moving_average(final_returns,
                                    title="Return history",
                                    ylabel="Return",
                                    output_path=out_path + "returns.png")
                plt.plot(self.return_records)
                plot(title="Return records",
                     xlabel="Episode",
                     ylabel="Return",
                     output_path=out_path + "return_records.png")

            # Plot win-loss-ratio line chart
            if self.win_relevant:
                plot_moving_average(self.get_wins(),
                                    title="Win-Raio",
                                    ylabel="Win proportion",
                                    output_path=out_path + "wins.png")

    def log_extreme_losses(self, individual_losses, trans_ids, predictions, targets, memory, env, logger):
        moving_avg_loss = get_moving_avg_val(self.get_train_losses(), WINDOW_SIZES_LOSS[0])
        if moving_avg_loss > 0:
            extreme_loss = individual_losses > (moving_avg_loss * EXTREME_LOSS_FACTOR)
            if np.any(extreme_loss):
                print(yellow("Extreme loss encountered!"))
                for trans_id, loss, prediction, target in zip(trans_ids[extreme_loss],
                                                              individual_losses[extreme_loss],
                                                              predictions[extreme_loss],
                                                              targets[extreme_loss]):
                    logger.log_extreme_loss(self.get_train_cycle(), loss, moving_avg_loss,
                                            str(prediction), str(target),
                                            memory.get_trans_text(trans_id, env) + "\n")


def get_moving_avg_val(values, window_size):
    """Computes a single, most recent moving average value given a window size."""
    if len(values) >= window_size:
        avg_val = np.average(values[-window_size:])
    else:
        avg_val = np.nan
    return avg_val


def get_moving_avg_lst(values, n):
    """Computes the moving average with window size n of a list of numbers. The output list
    is padded at the beginning."""
    mov_avg = np.cumsum(values, dtype=float)
    mov_avg[n:] = mov_avg[n:] - mov_avg[:-n]
    mov_avg = mov_avg[n - 1:] / n
    mov_avg = np.insert(mov_avg, 0, int(n / 2) * [None])
    return mov_avg


def plot_moving_average(values, title, ylabel, output_path, window_sizes=WINDOW_SIZES_EPISODES, xlabel="Episode",
                        logarithmic=False, validation_values=None, validation_period=None, show=False):
    add_moving_avg_plot(values, window_sizes[0], 'silver')
    add_moving_avg_plot(values, window_sizes[1], 'black')
    add_moving_avg_plot(values, window_sizes[2], '#009d81')

    if validation_values is not None and validation_period is not None:
        add_validation_plot(validation_values, validation_period)

    plot(title, xlabel, ylabel, output_path, True, logarithmic, show)


def add_moving_avg_plot(values, window_size, color=None, label=None):
    if label is None:
        label = "Moving average %d" % window_size

    if len(values) > window_size:
        mov_avg_ret = get_moving_avg_lst(values, window_size)
        plt.plot(mov_avg_ret, label=label, c=color)


def compare_statistics(model_names, env, labels=None, cut_at_episode=None, cut_at_train_cycle=None):
    """Takes a list of model names, retrieves their statistics and plots them."""
    base_path = "out/%s/" % env.NAME
    returns = []
    scores = []
    times = []
    losses = []
    wins = []
    return_records = []
    score_records = []
    time_records = []
    learning_rates = []

    # Gather multiple statistics
    for model_name in model_names:
        in_path = base_path + model_name + "/"
        stats = Statistics(env)
        stats.load(in_path)

        if cut_at_episode is None:
            cut_at_episode = stats.get_length()
        if cut_at_train_cycle is None:
            cut_at_train_cycle = stats.get_train_cycle()

        returns += [stats.get_final_returns()[:cut_at_episode]]
        scores += [stats.get_final_scores()[:cut_at_episode]]
        times += [stats.get_final_times()[:cut_at_episode]]
        losses += [stats.get_train_losses()[:cut_at_train_cycle]]
        wins += [stats.get_wins()[:cut_at_episode]]
        return_records += [stats.get_return_records()[:cut_at_episode]]
        score_records += [stats.get_score_records()[:cut_at_episode]]
        time_records += [stats.get_time_records()[:cut_at_episode]]
        learning_rates += [stats.get_learning_rates()[:cut_at_train_cycle]]

    # (Re-)create output folder
    if os.path.exists("out/comparison_plots"):
        shutil.rmtree("out/comparison_plots")
    os.mkdir("out/comparison_plots")

    if labels is None:
        labels = model_names

    # Generate all (relevant) comparison plots
    plot_comparison(out_path="out/comparison_plots/returns.png", comparison_values=returns, labels=labels,
                    title="Return history comparison", ylabel="Return", window_size=WINDOW_SIZES_EPISODES[1])
    plot_comparison(out_path="out/comparison_plots/return_records.png", comparison_values=return_records, labels=labels,
                    title="Return records comparison", ylabel="Return")

    plot_comparison(out_path="out/comparison_plots/scores.png", comparison_values=scores, labels=labels,
                    title="Score history comparison", ylabel="Score", window_size=WINDOW_SIZES_EPISODES[1])
    plot_comparison(out_path="out/comparison_plots/log_scores.png", comparison_values=scores, labels=labels,
                    title="Score history comparison", ylabel="Score", window_size=WINDOW_SIZES_EPISODES[1],
                    logarithmic=True)
    plot_comparison(out_path="out/comparison_plots/score_records.png", comparison_values=score_records, labels=labels,
                    title="Score records comparison", ylabel="Score")

    if env.TIME_RELEVANT:
        plot_comparison(out_path="out/comparison_plots/times.png", comparison_values=times, labels=labels,
                        title="Episode length history comparison", ylabel="Time (game ticks)",
                        window_size=WINDOW_SIZES_EPISODES[1])
        plot_comparison(out_path="out/comparison_plots/log_times.png", comparison_values=times, labels=labels,
                        title="Episode length history comparison", ylabel="Time (game ticks)",
                        window_size=WINDOW_SIZES_EPISODES[1], logarithmic=True)
        plot_comparison(out_path="out/comparison_plots/time_records.png", comparison_values=time_records,
                        labels=labels, title="Time records comparison", ylabel="Time (game ticks)")

    if env.WINS_RELEVANT:
        plot_comparison(out_path="out/comparison_plots/wins.png", comparison_values=wins, labels=labels,
                        title="Win-Ratio", ylabel="Win proportion", window_size=WINDOW_SIZES_EPISODES[1])

    plot_comparison(out_path="out/comparison_plots/losses.png", comparison_values=losses, labels=labels,
                    title="Training loss history comparison", ylabel="Loss", xlabel="Train cycle", logarithmic=True,
                    window_size=WINDOW_SIZES_LOSS[1])

    plot_comparison(out_path="out/comparison_plots/learning_rates.png", comparison_values=learning_rates, labels=labels,
                    title="Learning rates comparison", ylabel="Learning rate", xlabel="Train cycle")


def plot_comparison(out_path, comparison_values, labels, title, ylabel, window_size=None, xlabel="Episode",
                    logarithmic=False):
    if window_size is not None:
        for values, label in zip(comparison_values, labels):
            add_moving_avg_plot(values, window_size, label=label)
    else:
        for values, label in zip(comparison_values, labels):
            plt.plot(values, label=label)

    plot(title, xlabel, ylabel, out_path, True, logarithmic)


def add_validation_plot(validation_values, validation_period):
    n = len(validation_values)
    if n > 0:
        x = list(range(0, n * validation_period, validation_period))
        plt.plot(x, validation_values, label="Validation", c="orange")
