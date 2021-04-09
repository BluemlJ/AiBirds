import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time

from src.utils.utils import orange, yellow, red, bold, plot, user_agrees_to, sec2hhmmss, num2text, \
    data2pickle, pickle2data
from src.envs.env import Environment
from src.utils.logger import Logger
from src.utils.mem import ReplayMemory


class Statistics:
    # General stat axes
    TRANSITION = 0
    SECONDS = 1  # wall-clock time

    # Episode stat axes
    RETURN = 2
    SCORE = 3
    TIME = 4
    WIN = 5
    RETURN_RECORD = 6
    SCORE_RECORD = 7
    TIME_RECORD = 8

    # Train cycle stat axes
    LOSS = 2
    LEARNING_RATE = 3

    # Other constants
    WINDOW_SIZES_EPISODES = (500, 5000, 20000)
    WINDOW_SIZES_CYCLES = (10, 50, 200)
    EXTREME_LOSS_FACTOR = 500

    def __init__(self, env_type: Environment = None, env: Environment = None,
                 memory: ReplayMemory = None, log_path=None):
        self.env_type = env_type
        self.env = env
        self.memory = memory
        if log_path is not None:
            self.logger = Logger(log_path, self.EXTREME_LOSS_FACTOR, self.WINDOW_SIZES_CYCLES[0])
        else:
            self.logger = None

        self.episode_stats = np.zeros(shape=(100000, 9), dtype="float32")
        self.cycle_stats = np.zeros(shape=(5000, 4), dtype="float32")

        self.episode_ptr = 0
        self.cycle_ptr = 0

        self.ma_score_record = 0
        self.current_run_episode_no = 0
        self.continue_training = False

        self.total_timer = 0
        self.computation_timer = 0
        self.timer_started = False

    def _enlarge_episode_stats(self, incr_size=None, to_size=None):
        self.episode_stats = enlarge_stats(self.episode_stats, incr_size, to_size)

    def _enlarge_cycle_stats(self, incr_size=None, to_size=None):
        self.cycle_stats = enlarge_stats(self.cycle_stats, incr_size, to_size)

    def start_timer(self):
        self.total_timer = time.time() - self.total_timer
        self.computation_timer = time.time()
        self.timer_started = True

    def denote_episode_stats(self, ret, score, t, win):
        assert self.timer_started

        if self.episode_ptr == len(self.episode_stats):
            self._enlarge_episode_stats(100000)

        trans = self.get_current_transition() + t
        seconds = time.time() - self.total_timer
        ret_rec, score_rec, time_rec = self.get_records()
        new_return_record = ret_rec < ret
        ret_rec = max(ret_rec, ret)
        score_rec = max(score_rec, score)
        time_rec = max(time_rec, t)
        self.episode_stats[self.episode_ptr] = [trans, seconds, ret, score, t, win,
                                                ret_rec, score_rec, time_rec]
        self.episode_ptr += 1
        self.current_run_episode_no += 1

        if new_return_record and self.memory is not None and \
                self.logger is not None and self.env is not None:
            transition_text = self.memory.get_trans_text(self.memory.get_length() - 1, self.env)
            self.logger.log_new_record(ret, transition_text)

    def denote_learning_stats(self, loss, individual_losses, learning_rate, trans_ids, predictions,
                              targets, env):
        assert self.timer_started

        if self.cycle_ptr == len(self.cycle_stats):
            self._enlarge_cycle_stats(5000)

        trans = self.get_current_transition()
        seconds = time.time() - self.total_timer
        self.cycle_stats[self.cycle_ptr] = [trans, seconds, loss, learning_rate]
        self.cycle_ptr += 1

        self.log_extreme_losses(individual_losses, trans_ids, predictions, targets, env)

    def get_num_episodes(self):
        return self.episode_ptr

    def get_num_cycles(self):
        return self.cycle_ptr

    def get_current_transition(self):
        if self.get_num_episodes() == 0:
            return 0
        else:
            return self.episode_stats[self.episode_ptr - 1, self.TRANSITION]

    def get_episode_transitions(self):
        return self.episode_stats[:self.episode_ptr, self.TRANSITION].astype("int")

    def get_episode_seconds(self):
        """Returns wall-clock time in seconds."""
        return self.episode_stats[:self.episode_ptr, self.SECONDS].astype("int")

    def get_episode_hours(self):
        return self.get_episode_seconds() / 3600

    def get_returns(self):
        return self.episode_stats[:self.episode_ptr, self.RETURN]

    def get_scores(self):
        return self.episode_stats[:self.episode_ptr, self.SCORE].astype("int")

    def get_times(self):
        return self.episode_stats[:self.episode_ptr, self.TIME].astype("int")

    def get_wins(self):
        if self.env_type.WINS_RELEVANT:
            return self.episode_stats[:self.episode_ptr, self.WIN].astype("bool")
        else:
            return []

    def get_return_records(self):
        return self.episode_stats[:self.episode_ptr, self.RETURN_RECORD]

    def get_score_records(self):
        return self.episode_stats[:self.episode_ptr, self.SCORE_RECORD].astype("int")

    def get_time_records(self):
        if self.env_type.TIME_RELEVANT:
            return self.episode_stats[:self.episode_ptr, self.TIME_RECORD].astype("int")
        else:
            return []

    def get_cycle_transitions(self):
        return self.cycle_stats[:self.episode_ptr, self.TRANSITION].astype("int")

    def get_cycle_seconds(self):
        """Returns wall-clock time in seconds."""
        return self.cycle_stats[:self.episode_ptr, self.SECONDS].astype("int")

    def get_cycle_hours(self):
        return self.get_cycle_seconds() / 3600

    def get_losses(self):
        return self.cycle_stats[:self.cycle_ptr, self.LOSS]

    def get_learning_rates(self):
        return self.cycle_stats[:self.cycle_ptr, self.LEARNING_RATE]

    def get_current_score(self):
        if self.get_num_episodes() > 0:
            return int(self.episode_stats[self.episode_ptr - 1, self.SCORE])
        else:
            return np.nan

    def get_current_learning_rate(self):
        if self.get_num_cycles() > 0:
            return self.episode_stats[self.cycle_ptr - 1, self.LEARNING_RATE]
        else:
            return np.nan

    def get_records(self):
        if self.get_num_episodes() > 0:
            ret_rec = self.episode_stats[self.episode_ptr - 1, self.RETURN_RECORD]
            score_rec = int(self.episode_stats[self.episode_ptr - 1, self.SCORE_RECORD])
            time_rec = self.episode_stats[self.episode_ptr - 1, self.TIME_RECORD]
            return ret_rec, score_rec, time_rec
        else:
            return -np.inf, 0, 0

    def compute_records(self):
        num_episodes = self.get_num_episodes()

        if num_episodes == 0:
            return

        rec_axes = [self.RETURN_RECORD, self.SCORE_RECORD, self.TIME_RECORD]
        stat_axes = [self.RETURN, self.SCORE, self.TIME]
        records = self.episode_stats[0, stat_axes]

        for episode in range(num_episodes):
            records = np.max((records, self.episode_stats[episode, stat_axes]), axis=0)
            self.episode_stats[episode, rec_axes] = records

    def compute_transitions(self):
        num_episodes = self.get_num_episodes()

        if num_episodes == 0:
            return

        transition = 0
        for episode in range(num_episodes):
            transition += self.episode_stats[episode, self.TIME]
            self.episode_stats[episode, self.TRANSITION] = transition

    def get_stats_dict(self):
        stats_dict = {"episode_stats": self.episode_stats,
                      "cycle_stats": self.cycle_stats,
                      "episode_ptr": self.episode_ptr,
                      "cycle_ptr": self.cycle_ptr}
        return stats_dict

    def get_stats(self):
        return self.episode_stats[:self.episode_ptr], self.cycle_stats[:self.cycle_ptr]

    def set_stats(self, stats_dict):
        self.episode_stats = stats_dict["episode_stats"]
        self.cycle_stats = stats_dict["cycle_stats"]
        self.episode_ptr = stats_dict["episode_ptr"]
        self.cycle_ptr = stats_dict["cycle_ptr"]
        self.total_timer = self.episode_stats[self.episode_ptr - 1, self.SECONDS]

    def set_stats_old(self, stats_dict):
        # Read in episode stats
        num_episodes = len(stats_dict["returns"])
        self.episode_ptr = num_episodes
        if num_episodes > len(self.episode_stats):
            self._enlarge_episode_stats(to_size=num_episodes)

        for key, i in [("returns", self.RETURN), ("scores", self.SCORE),
                       ("times", self.TIME), ("wins", self.WIN)]:
            values = stats_dict[key]
            self.episode_stats[:len(values), i] = values

        # Read in cycle stats
        num_cycles = len(stats_dict["losses"])
        self.cycle_ptr = num_cycles
        if num_cycles > len(self.cycle_stats):
            self._enlarge_cycle_stats(to_size=num_cycles)

        for key, i in [("losses", self.LOSS), ("learning_rates", self.LEARNING_RATE)]:
            values = stats_dict[key]
            self.cycle_stats[:len(values), i] = values

        self.compute_records()
        self.compute_transitions()

    def save(self, out_path):
        file_path = out_path + "stats.pckl"
        data2pickle(self.get_stats_dict(), file_path)

    def load(self, in_path):
        file_path = in_path + "stats.pckl"
        data = pickle2data(file_path)
        if data is None:
            print(orange("No statistics loaded. Continuing without statistics."))
        else:
            try:
                self.set_stats(data)
                print("Successfully loaded stats in new format from '%s'." % file_path)
            except Exception as e:
                print(orange("Failed to load stats in new format."))
                print(e)
                try:
                    print("Trying to load stats in old format and convert...")
                    self.set_stats_old(data)
                    print("Successfully loaded stats in old format from '%s'." % file_path)
                    self.save(in_path)
                    print("Converted and saved in new format.")
                except Exception as e:
                    print(red("Failed to load statistics from '%s'! Skipping this step." % file_path))
                    print(e)

    def progress_stagnates(self):
        """Returns true if loss did not change much and score did not increase."""
        return self.loss_stagnates() and self.score_stagnates()

    def loss_stagnates(self):
        if self.get_num_cycles() >= 1000:
            losses = self.get_losses()
            magnitude = np.average(losses[-1000:])
            long_term_stagnation = np.abs(np.average(losses[-1000:-500]) -
                                          np.average(losses[-500:])) < 0.005 * magnitude
            short_term_stagnation = np.abs(np.average(losses[-400:-200]) -
                                           np.average(losses[-200:])) < 0.005 * magnitude
            return long_term_stagnation and short_term_stagnation
        else:
            return False

    def score_stagnates(self):
        """Returns true if score improvement was low. More precisely, returns True if the 1000 MA score did improve
        by less than 5% during the last 4000 episodes."""
        if self.get_num_episodes() >= 5000:
            scores = self.get_scores()
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

    def print_stats(self, par_step, total_par_steps, num_envs, print_stats_period, epsilon):
        assert self.timer_started
        comp_time = time.time() - self.computation_timer
        total_time = time.time() - self.total_timer

        ma_episode_size = self.WINDOW_SIZES_EPISODES[1]
        ma_cycle_size = self.WINDOW_SIZES_CYCLES[1]

        # avg_return = get_moving_avg(self.get_final_returns(), ma_episode_size)
        ma_score = get_moving_avg_val(self.get_scores(), ma_episode_size)
        ma_loss = get_moving_avg_val(self.get_losses(), ma_cycle_size)
        self.ma_score_record = np.max((self.ma_score_record, ma_score))

        return_record, score_record, time_record = self.get_records()
        done_transitions = par_step * num_envs
        ma_text = num2text(ma_episode_size)

        stats_text = "\n" + bold("Parallel step %d/%d" % (par_step, total_par_steps)) + \
                     "\n   Completed episodes:        %s" % num2text(self.get_num_episodes()) + \
                     "\n   Transitions done:          %s" % num2text(done_transitions) + \
                     "\n   Epsilon:                   %.3f" % epsilon.get_value() + \
                     "\n   " + "{:27s}".format("Score (%s MA):" % ma_text) + ("%.1f" % ma_score) + \
                     "\n   Score record:              %.0f" % score_record

        if self.env_type.WINS_RELEVANT:
            avg_wins = get_moving_avg_val(self.get_wins(), ma_episode_size)
            stats_text += \
                "\n   " + "{:27s}".format("Win-ratio (%s MA):" % ma_text) + ("%.2f" % avg_wins)

        if self.env_type.TIME_RELEVANT:
            avg_time = get_moving_avg_val(self.get_times(), ma_episode_size)
            stats_text += \
                "\n   " + "{:27s}".format("Time (%s MA):" % ma_text) + ("%.1f" % avg_time) + \
                "\n   Time record:               %.0f" % time_record

        stats_text += "\n   " + "{:27s}".format("Loss (%s MA):" % num2text(ma_cycle_size)) + ("%.4f" % ma_loss) + \
                      "\n   Learning rate:             %.6f" % self.get_current_learning_rate() + \
                      "\n------" \
                      "\n   " + "{:27s}".format("Comp time (last %d):" % print_stats_period) + \
                      "%d s" % np.round(comp_time) + \
                      "\n   Total comp time:           " + sec2hhmmss(total_time)

        print(stats_text)
        if self.logger is not None:
            self.logger.log_step_statistics(stats_text + "\n")

        if self.score_crashed(ma_score) and not self.continue_training:
            question = orange("Score crashed! %s MA score dropped by 95 %%. "
                              "Still want to continue training? (y/n)" % ma_text)
            if not user_agrees_to(question):
                quit()
            else:
                self.continue_training = True

        self.computation_timer = time.time()

    def plot_stats(self, out_path):
        if self.get_num_episodes() >= 200:  # TODO
            self.plot_scores(out_path)
            self.plot_returns(out_path)
            self.plot_times(out_path)
            if self.env_type.WINS_RELEVANT:
                self.plot_wins(out_path)
            self.plot_learning_rates(out_path)
            self.plot_loss(out_path)
            self.plot_priorities(out_path)

    def plot_scores(self, out_path):
        final_scores = self.get_scores()
        if len(final_scores) > 0:
            plot_moving_average(final_scores,
                                title="Score history",
                                ylabel="Score",
                                window_sizes=self.WINDOW_SIZES_EPISODES,
                                output_path=out_path + "scores.png",
                                show=True)
            plt.plot(self.get_score_records())
            plot(title="Score records",
                 x_label="Episode",
                 y_label="Score",
                 out_path=out_path + "score_records.png")

    def plot_returns(self, out_path):
        final_returns = self.get_returns()
        if len(final_returns) > 0:
            plot_moving_average(final_returns,
                                title="Return history",
                                ylabel="Return",
                                window_sizes=self.WINDOW_SIZES_EPISODES,
                                output_path=out_path + "returns.png")
            plt.plot(self.get_return_records())
            plot(title="Return records",
                 x_label="Episode",
                 y_label="Return",
                 out_path=out_path + "return_records.png")

    def plot_times(self, out_path):
        if self.env_type.TIME_RELEVANT:
            plot_moving_average(self.get_times(),
                                title="Episode length history",
                                ylabel="Time",
                                window_sizes=self.WINDOW_SIZES_EPISODES,
                                output_path=out_path + "times.png")
            plt.plot(self.get_time_records())
            plot(title="Episode length records",
                 x_label="Episode",
                 y_label="Time (game ticks)",
                 out_path=out_path + "time_records.png")

    def plot_wins(self, out_path):
        plot_moving_average(self.get_wins(),
                            title="Win-Raio",
                            ylabel="Win proportion",
                            window_sizes=self.WINDOW_SIZES_EPISODES,
                            output_path=out_path + "wins.png")

    def plot_learning_rates(self, out_path):
        if self.get_current_learning_rate() is not np.nan:
            plt.plot(self.get_learning_rates())
            plt.ylim(bottom=0)
            plot(title="Learning rate history",
                 x_label="Train cycle", y_label="Learning rate",
                 out_path=out_path + "learning_rates.png",
                 legend=False,
                 logarithmic=False)

    def plot_loss(self, out_path):
        if self.get_num_cycles() > self.WINDOW_SIZES_CYCLES[0]:
            plot_moving_average(self.get_losses(),
                                title="Training loss history",
                                window_sizes=self.WINDOW_SIZES_CYCLES,
                                ylabel="Loss", xlabel="Train cycle",
                                output_path=out_path + "loss.png",
                                logarithmic=True)

    def plot_priorities(self, out_path):
        if self.memory is not None:
            plt.hist(self.memory.get_priorities(), range=None, bins=100)
            plot(title="Transition priorities in experience set",
                 x_label="Priority value",
                 y_label="Number of transitions",
                 out_path=out_path + "priorities.png")

    def log_extreme_losses(self, individual_losses, trans_ids, predictions, targets, env):
        if self.logger is None:
            return
        moving_avg_loss = get_moving_avg_val(self.get_losses(), self.WINDOW_SIZES_CYCLES[0])
        if moving_avg_loss > 0:
            extreme_loss = individual_losses > (moving_avg_loss * self.EXTREME_LOSS_FACTOR)
            if np.any(extreme_loss):
                print(yellow("Extreme loss encountered!"))
                for trans_id, loss, prediction, target in zip(trans_ids[extreme_loss],
                                                              individual_losses[extreme_loss],
                                                              predictions[extreme_loss],
                                                              targets[extreme_loss]):
                    self.logger.log_extreme_loss(self.get_num_cycles(), loss, moving_avg_loss,
                                                 str(prediction), str(target),
                                                 self.memory.get_trans_text(trans_id, env) + "\n")


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


def plot_moving_average(values, title, ylabel, output_path, window_sizes, xlabel="Episode",
                        logarithmic=False, validation_values=None, validation_period=None, show=False):
    add_moving_avg_plot(values, window_sizes[0], 'silver')
    add_moving_avg_plot(values, window_sizes[1], 'black')
    add_moving_avg_plot(values, window_sizes[2], '#009d81')

    if validation_values is not None and validation_period is not None:
        add_validation_plot(validation_values, validation_period)

    plot(title, xlabel, ylabel, output_path, True, logarithmic, show)


def add_moving_avg_plot(y_values, window_size, x_values=None, color=None, label=None):
    if label is None:
        label = "Moving average %d" % window_size

    if len(y_values) > window_size:
        y_values_ma = get_moving_avg_lst(y_values, window_size)
        if x_values is None:
            x_values = range(len(y_values_ma))
        else:
            x_values = x_values[:len(y_values_ma)]
        plt.plot(x_values, y_values_ma, label=label, c=color)


def compare_statistics(model_names, env_type, labels=None, cut_at_episode=None, cut_at_cycle=None):
    """Takes a list of model names, retrieves their statistics and plots them."""
    base_path = "out/%s/" % env_type.NAME

    stats_collection = []
    max_num_episodes = 0
    max_num_cycles = 0

    # Retrieve all specified statistics
    for model_name in model_names:
        in_path = base_path + model_name + "/"
        stats = Statistics(env_type)
        stats.load(in_path)
        max_num_episodes = max(max_num_episodes, stats.get_num_episodes())
        max_num_cycles = max(max_num_cycles, stats.get_num_cycles())
        stats_collection += [stats]

    if labels is None:
        labels = model_names

    s = Statistics(env_type)
    ma_ws_episodes = s.WINDOW_SIZES_EPISODES[1]
    ma_ws_cycles = s.WINDOW_SIZES_CYCLES[1]

    if cut_at_episode is None:
        cut_at_episode = max_num_episodes
    if cut_at_cycle is None:
        cut_at_cycle = max_num_cycles

    out_path = "out/comparison_plots/"

    plot_everything(stats_collection=stats_collection,
                    labels=labels,
                    env_type=env_type,
                    ma_ws_episodes=ma_ws_episodes,
                    ma_ws_cycles=ma_ws_cycles,
                    out_path=out_path,
                    cut_at_episode=cut_at_episode,
                    cut_at_cycle=cut_at_cycle)


def plot_everything(out_path, env_type: Environment, **kwargs):
    # (Re-)create output folder
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # Episode stats
    for domain, x_label in zip(["episodes", "transitions", "wall-clock_time"],
                               ["Episode", "Transition", "Wall-clock time (h)"]):
        sub_out_path = out_path + domain + "/"
        os.mkdir(sub_out_path)
        compare_returns(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)
        compare_scores(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)
        if env_type.TIME_RELEVANT:
            compare_times(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)
        if env_type.WINS_RELEVANT:
            compare_wins(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)

    # Cycle stats
    for domain, x_label in zip(["cycles", "transitions", "wall-clock_time"],
                               ["Cycle", "Transition", "Wall-clock time (h)"]):
        sub_out_path = out_path + domain + "/"
        if not os.path.exists(sub_out_path):
            os.mkdir(sub_out_path)
        compare_losses(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)
        compare_learning_rates(domain=domain, x_label=x_label, out_path=sub_out_path, **kwargs)


def compare_returns(ma_ws_episodes, **kwargs):
    compare_values_on_domain(name="returns",
                             getter_handle=Statistics.get_returns,
                             orig_domain="episodes",
                             title="Return history comparison",
                             y_label="Return",
                             ma_ws_episodes=ma_ws_episodes,
                             logarithmic=False,
                             **kwargs)
    compare_values_on_domain(name="return_records",
                             getter_handle=Statistics.get_return_records,
                             orig_domain="episodes",
                             title="Return records comparison",
                             y_label="Return",
                             ma_ws_episodes=None,
                             logarithmic=False,
                             **kwargs)


def compare_scores(ma_ws_episodes, **kwargs):
    compare_values_on_domain(name="scores",
                             getter_handle=Statistics.get_scores,
                             orig_domain="episodes",
                             title="Score history comparison",
                             y_label="Score",
                             ma_ws_episodes=ma_ws_episodes,
                             logarithmic=True,
                             **kwargs)
    compare_values_on_domain(name="score_records",
                             getter_handle=Statistics.get_score_records,
                             orig_domain="episodes",
                             title="Score records comparison",
                             y_label="Score",
                             ma_ws_episodes=None,
                             logarithmic=True,
                             **kwargs)


def compare_times(ma_ws_episodes, **kwargs):
    compare_values_on_domain(name="times",
                             getter_handle=Statistics.get_times,
                             orig_domain="episodes",
                             title="Time history comparison",
                             y_label="Episode length",
                             ma_ws_episodes=ma_ws_episodes,
                             logarithmic=True,
                             **kwargs)
    compare_values_on_domain(name="time_records",
                             getter_handle=Statistics.get_score_records,
                             orig_domain="episodes",
                             title="Time records comparison",
                             y_label="Episode length",
                             ma_ws_episodes=None,
                             logarithmic=True,
                             **kwargs)


def compare_wins(**kwargs):
    compare_values_on_domain(name="wins",
                             getter_handle=Statistics.get_wins,
                             orig_domain="episodes",
                             title="Win-ratio history comparison",
                             y_label="Win proportion",
                             logarithmic=False,
                             **kwargs)


def compare_losses(**kwargs):
    compare_values_on_domain(name="losses",
                             getter_handle=Statistics.get_losses,
                             orig_domain="cycles",
                             title="Loss history comparison",
                             y_label="Loss",
                             logarithmic=True,
                             **kwargs)


def compare_learning_rates(**kwargs):
    compare_values_on_domain(name="learning-rates",
                             getter_handle=Statistics.get_learning_rates,
                             orig_domain="cycles",
                             title="Learning rate history comparison",
                             y_label="Learning rate",
                             logarithmic=False,
                             **kwargs)


def compare_values_on_domain(name, stats_collection, labels, getter_handle, domain, orig_domain, title,
                             x_label, y_label, ma_ws_episodes, ma_ws_cycles, out_path, logarithmic,
                             cut_at_episode, cut_at_cycle):
    if orig_domain == "episodes":
        get_domain = get_episodes_domain
        cut_at = cut_at_episode
        ma_window_size = ma_ws_episodes
    elif orig_domain == "cycles":
        get_domain = get_cycles_domain
        cut_at = cut_at_cycle
        ma_window_size = ma_ws_cycles
    else:
        raise ValueError("Invalid original domain given.")

    if ma_window_size is not None:
        y_label += " (%s MA)" % num2text(ma_window_size)

    for stats, label in zip(stats_collection, labels):
        x_values = get_domain(stats, domain, cut_at)
        y_values = getter_handle(stats)[:cut_at]
        if ma_window_size is not None:
            add_moving_avg_plot(x_values=x_values, y_values=y_values,
                                window_size=ma_window_size, label=label)
        else:
            if x_values is not None:
                plt.plot(x_values, y_values, label=label)
            else:
                plt.plot(y_values, label=label)
    keep = logarithmic
    plot(title=title, x_label=x_label, y_label=y_label, out_path=out_path + name + ".png",
         legend=True, logarithmic=False, keep=keep)
    if logarithmic:
        plot(title=title, x_label=x_label, y_label=y_label, out_path=out_path + "log_" + name + ".png",
             legend=True, logarithmic=True, keep=False)


def get_episodes_domain(stats, domain, cut_at_episode):
    if domain == "episodes":
        return None
    elif domain == "transitions":
        return stats.get_episode_transitions()[:cut_at_episode]
    elif domain == "wall-clock_time":
        return stats.get_episode_hours()[:cut_at_episode]
    else:
        raise ValueError("Invalid domain name provided.")


def get_cycles_domain(stats, domain, cut_at_cycle):
    if domain == "cycles":
        return None
    elif domain == "transitions":
        return stats.get_cycle_transitions()[:cut_at_cycle]
    elif domain == "wall-clock_time":
        return stats.get_cycle_hours()[:cut_at_cycle]
    else:
        raise ValueError("Invalid domain name provided.")


def plot_comparison(out_path, y_values, labels, title, y_label, x_values=None, window_size=None,
                    x_label="Episode", logarithmic=False):
    if x_values is None:
        x_values = len(y_values) * [None]

    if window_size is not None:
        for x, y, label in zip(x_values, y_values, labels):
            add_moving_avg_plot(y, window_size, x_values=x, label=label)
    else:
        for x, y, label in zip(x_values, y_values, labels):
            if x is None:
                x = range(len(y))
                plt.plot(x, y, label=label)

    plot(title, x_label, y_label, out_path, True, logarithmic)


def add_validation_plot(validation_values, validation_period):
    n = len(validation_values)
    if n > 0:
        x = list(range(0, n * validation_period, validation_period))
        plt.plot(x, validation_values, label="Validation", c="orange")


def enlarge_stats(stat_array, incr_size, to_size):
    shape = stat_array.shape
    if to_size is not None:
        assert to_size > shape[0]
        new_size = to_size
    else:
        assert incr_size > 0
        new_size = shape[0] + incr_size
    new_episode_stats = np.zeros(shape=(new_size, shape[1]))
    new_episode_stats[:shape[0]] = stat_array
    return new_episode_stats
