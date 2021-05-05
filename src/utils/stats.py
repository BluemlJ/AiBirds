import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time

from src.utils.utils import finalize_plot, user_agrees_to, sec2hhmmss, num2text, \
    data2pickle, pickle2data
from src.envs.env import Environment
from src.utils.logger import Logger
from src.mem.mem import ReplayMemory
from src.utils.text_sty import orange, red, bold, print_info
from skimage.measure import block_reduce

# For comparison plots
PLOT_STEP_NO = 10000
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


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
    WINDOW_SIZE_EPISODES = 500
    WINDOW_SIZE_CYCLES = 10
    EXTREME_LOSS_FACTOR = 1000

    def __init__(self, env_type: Environment = None, env: Environment = None, log_path=None):
        assert env_type is not None or env is not None
        self.env_type = type(env) if env_type is None else env_type
        self.env = env
        if log_path is not None:
            self.logger = Logger(log_path, self.EXTREME_LOSS_FACTOR, self.WINDOW_SIZE_CYCLES)
        else:
            self.logger = None

        self.episode_stats = np.zeros(shape=(100000, 9), dtype="float32")
        self.cycle_stats = np.zeros(shape=(5000, 4), dtype="float32")

        self.episode_ptr = 0
        self.cycle_ptr = 0

        self.ma_score_record = 0
        self.current_run_trans_no = 0
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

    def denote_episode_stats(self, ret, score, t, win, env_id, memory):
        assert self.timer_started

        if self.episode_ptr == len(self.episode_stats):
            self._enlarge_episode_stats(100000)

        trans = self.get_current_transition() + t
        seconds = time.time() - self.total_timer
        ret_rec, score_rec, time_rec = self.get_records()
        new_return_record = ret_rec < ret
        ret_rec = max(ret, ret_rec)
        score_rec = max(score, score_rec)
        time_rec = max(t, time_rec)
        self.episode_stats[self.episode_ptr] = [trans, seconds, ret, score, t, win,
                                                ret_rec, score_rec, time_rec]
        self.episode_ptr += 1
        self.current_run_trans_no += t

        if new_return_record and self.logger is not None and self.env is not None:
            trans_id = memory.idx2id([memory.trans_buf.stack_ptr, env_id])
            transition_text = memory.get_trans_text(trans_id, self.env)
            self.logger.log_new_record(ret, transition_text)

    def denote_learning_stats(self, loss, individual_losses, learning_rate, trans_ids, predictions,
                              targets, env, memory):
        assert self.timer_started

        if self.cycle_ptr == len(self.cycle_stats):
            self._enlarge_cycle_stats(5000)

        trans = self.get_current_transition()
        seconds = time.time() - self.total_timer
        self.cycle_stats[self.cycle_ptr] = [trans, seconds, loss, learning_rate]
        self.cycle_ptr += 1

        self.log_extreme_losses(individual_losses, trans_ids, predictions, targets, env, memory)

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
        return self.cycle_stats[:self.cycle_ptr, self.TRANSITION].astype("int")

    def get_cycle_seconds(self):
        """Returns wall-clock time in seconds."""
        return self.cycle_stats[:self.cycle_ptr, self.SECONDS].astype("int")

    def get_cycle_hours(self):
        return self.get_cycle_seconds() / 3600

    def get_losses(self):
        return self.cycle_stats[:self.cycle_ptr, self.LOSS]

    def get_learning_rates(self):
        return self.cycle_stats[:self.cycle_ptr, self.LEARNING_RATE]

    def get_current_score(self):
        if self.get_num_episodes() > 0:
            return self.get_scores()[-1]
        else:
            return np.nan

    def get_current_learning_rate(self):
        if self.get_num_cycles() > 0 and self.current_run_trans_no > 0:
            return self.get_learning_rates()[-1]
        else:
            return np.nan

    def get_records(self):
        if self.get_num_episodes() > 0:
            ret_rec = self.episode_stats[self.episode_ptr - 1, self.RETURN_RECORD]
            score_rec = int(self.episode_stats[self.episode_ptr - 1, self.SCORE_RECORD])
            time_rec = self.episode_stats[self.episode_ptr - 1, self.TIME_RECORD]
            return ret_rec, score_rec, time_rec
        else:
            return np.nan, np.nan, np.nan

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

    def print_stats(self, par_step, total_par_steps, print_stats_period, num_par_envs,
                    epsilon: int, ma_episode_size):
        assert self.timer_started
        comp_time = time.time() - self.computation_timer
        total_time = time.time() - self.total_timer

        ma_cycle_size = self.WINDOW_SIZE_CYCLES

        ma_score = get_moving_avg_val(self.get_scores(), ma_episode_size)
        ma_loss = get_moving_avg_val(self.get_losses(), ma_cycle_size)
        self.ma_score_record = np.max((self.ma_score_record, ma_score))

        return_record, score_record, time_record = self.get_records()
        ma_text = num2text(ma_episode_size)

        ep = num2text(self.get_num_episodes())
        trans = num2text(par_step * num_par_envs)
        cyc = num2text(self.get_num_cycles())

        stats_text = "\n" + bold("Parallel step %d/%d" % (par_step, total_par_steps)) + \
                     "\n   # Ep. | Trans. | Cyc.:     %s | %s | %s" % (ep, trans, cyc) + \
                     "\n   Epsilon:                   %.3f" % epsilon + \
                     "\n   " + "{:27s}".format("Score (%s MA | record):" % ma_text) + \
                     ("%.1f | %.0f" % (ma_score, score_record))

        if self.env_type.WINS_RELEVANT:
            ma_wins = get_moving_avg_val(self.get_wins(), ma_episode_size)
            stats_text += \
                "\n   " + "{:27s}".format("Win-ratio (%s MA):" % ma_text) + ("%.2f" % ma_wins)

        if self.env_type.TIME_RELEVANT:
            ma_time = get_moving_avg_val(self.get_times(), ma_episode_size)
            stats_text += \
                "\n   " + "{:27s}".format("Time (%s MA | record):" % ma_text) + \
                ("%.1f | %d" % (ma_time, time_record))

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

    def plot_stats(self, memory: ReplayMemory, out_path):
        self.plot_scores(out_path)
        self.plot_returns(out_path)
        if self.env_type.TIME_RELEVANT:
            self.plot_times(out_path)
        if self.env_type.WINS_RELEVANT:
            self.plot_wins(out_path)
        self.plot_learning_rates(out_path)
        self.plot_loss(out_path)
        priorities = memory.get_priorities().flatten()
        self.plot_priorities(priorities, out_path)
        self.plot_score_dist(out_path)

    def plot_real_time(self, y, y_records, title, y_label, out_path, show):
        transitions = self.get_episode_transitions()
        add_charts(x_lsts=[transitions],
                   y_lsts=[y],
                   labels=["Moving average"],
                   mov_avg=True,
                   interp=True,
                   plot_confidence=True,
                   confidence_labels=["90% confidence"])
        plt.plot(transitions, y_records, label="Records", color=DEFAULT_COLORS[1])
        finalize_plot(title=title,
                      x_label="Transition",
                      y_label=y_label,
                      out_path=out_path,
                      legend=True,
                      show=show)

    def plot_scores(self, out_path):
        scores = self.get_scores()
        if len(scores) > 0:
            self.plot_real_time(y=scores, y_records=self.get_score_records(), title="Score history",
                                y_label="Score", out_path=out_path + "scores.png", show=True)

    def plot_returns(self, out_path):
        returns = self.get_returns()
        if len(returns) > 0:
            self.plot_real_time(y=returns, y_records=self.get_return_records(), title="Return history",
                                y_label="Return", out_path=out_path + "returns.png", show=False)

    def plot_times(self, out_path):
        times = self.get_times()
        if len(times) > 0:
            self.plot_real_time(y=times, y_records=self.get_time_records(), title="Time history",
                                y_label="Episode length", out_path=out_path + "times.png", show=False)

    def plot_wins(self, out_path):
        plot_moving_average(self.get_wins(),
                            title="Win-Raio",
                            x_label="Episode",
                            y_label="Win proportion",
                            out_path=out_path + "wins.png")

    def plot_learning_rates(self, out_path):
        if self.get_current_learning_rate() is not np.nan:
            plt.plot(self.get_learning_rates())
            plt.ylim(bottom=0)
            finalize_plot(title="Learning rate history",
                          x_label="Train cycle", y_label="Learning rate",
                          out_path=out_path + "learning_rates.png",
                          legend=False,
                          logarithmic=False)

    def plot_loss(self, out_path):
        plot_moving_average(self.get_losses(),
                            title="Training loss history",
                            x_label="Train cycle", y_label="Loss",
                            out_path=out_path + "loss.png",
                            logarithmic=True)

    def plot_priorities(self, priorities, out_path):
        plt.hist(priorities, range=None, bins=100)
        finalize_plot(title="Transition priorities in experience set",
                      x_label="Priority value",
                      y_label="Number of transitions",
                      out_path=out_path + "priorities.png")

    def plot_score_dist(self, out_path):
        scores = self.get_scores()[-1000:]
        if len(scores) == 1000:
            n_groups = 5
            max_score, min_score = np.max(scores), np.min(scores)
            range_len = np.max(scores) - np.min(scores)
            if range_len >= 20:
                bin_size = np.ceil(range_len / 20)
            else:
                bin_size = 1
            bins = np.arange(min_score, max_score + 1, bin_size) - 0.5
            scores_grouped = np.flip(scores).reshape(n_groups, -1).T

            cmap = plt.cm.get_cmap('Blues_r', n_groups)
            colors = [cmap(val) for val in np.arange(0, 1, 0.2)]

            plt.hist(scores_grouped, range=None, bins=bins, histtype="barstacked", rwidth=0.7, color=colors)
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap), ticks=[0, 1])
            cbar.ax.set_xticklabels(['Old', 'Recent'])
            finalize_plot(title="Recent 1000 episode's score distribution",
                          x_label="Score",
                          y_label="Number of episodes",
                          out_path=out_path + "score_dist.png")

    def log_extreme_losses(self, individual_losses, trans_ids, predictions, targets, env, memory):
        if self.logger is None:
            return
        moving_avg_loss = get_moving_avg_val(self.get_losses(), self.WINDOW_SIZE_CYCLES)
        if moving_avg_loss > 0:
            extreme_loss = individual_losses > (moving_avg_loss * self.EXTREME_LOSS_FACTOR)
            if np.any(extreme_loss):
                print_info("Extreme loss encountered!")
                for trans_id, loss, prediction, target in zip(trans_ids[extreme_loss],
                                                              individual_losses[extreme_loss],
                                                              predictions[extreme_loss],
                                                              targets[extreme_loss]):
                    self.logger.log_extreme_loss(self.get_num_cycles(), loss, moving_avg_loss,
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
    is padded with NaN at the beginning and at the end to maintain length."""
    if len(values) < n:
        return np.array([])
    if n == 0:
        return values

    a = np.cumsum(values, dtype=float)
    a[n:] = a[n:] - a[:-n]
    a = a[n - 1:] / n
    mov_avg = np.empty(len(values))
    mov_avg[:] = np.nan
    mov_avg[n // 2:n // 2 + len(a)] = a
    return mov_avg


def get_bound(x, y, n, bound="upper"):
    if len(y) < n:
        return np.array([]), np.array([])
    if n == 0:
        return x, y

    # Ensure lengths of x and y are divisible by n
    overhang = len(y) % n
    if overhang:
        y = y[:-overhang]
        x = x[:-overhang]

    quantile = 0.05

    if bound == "upper":
        y = block_reduce(y, (n,), np.quantile, func_kwargs={"q": 1 - quantile})
    else:
        y = block_reduce(y, (n,), np.quantile, func_kwargs={"q": quantile})

    x = block_reduce(x, (n,), np.average)
    return x, y


def plot_moving_average(y, y_label, x=None, interp=False, plot_variance=False,
                        validation_values=None, validation_period=None, **kwargs):
    ma_ws = add_charts(x_lsts=[x], y_lsts=[y], mov_avg=True, interp=interp, plot_confidence=plot_variance)
    if ma_ws is not None:
        y_label += " (%s MA)" % num2text(ma_ws)

    if validation_values is not None and validation_period is not None:
        add_validation_plot(validation_values, validation_period)

    finalize_plot(y_label=y_label, **kwargs)


def add_charts(y_lsts, x_lsts=None, colors=None, labels=None, mov_avg=False,
               zero2nan=False, interp=False, plot_confidence=False, confidence_labels=None):
    """Adds one or multiple line chart(s) to an existing plot. Optionally converts into
    moving average (MA) if mov_avg=True. Automatically determines suitable MA windows size.
    Returns the used MA window size."""
    assert not interp or x_lsts is not None
    assert not plot_confidence or mov_avg

    if interp:
        max_len = PLOT_STEP_NO
        max_x = 0
        for x in x_lsts:
            if len(x) > 0:
                max_x = max(max_x, x[-1])
        step_size = max_x / max_len
    else:
        step_size = 1
        interp = False
        max_len = np.max([len(y) for y in y_lsts])

    ma_ws = max_len // 40

    for i, y in enumerate(y_lsts):
        label = labels[i] if labels is not None else None
        color = colors[i] if colors is not None else DEFAULT_COLORS[i]

        if len(y) <= 1:
            add_empty_plot(label=label, color=color)
            continue

        if interp:
            x, y = interpolate(x_lsts[i], y, step_size)
        else:
            x = None

        if mov_avg:
            y_ma = get_moving_avg_lst(y, ma_ws)

            if len(y_ma) == 0 and label is not None:
                add_empty_plot(label=label, color=color)
                continue

            if x is None:
                x = np.arange(len(y_ma))

            if zero2nan:
                y_ma[y_ma <= 0] = np.nan

            plt.plot(x, y_ma, label=label, color=color)

            if plot_confidence:
                var_label = confidence_labels[i] if confidence_labels is not None else None
                _, y_up = get_bound(x, y, ma_ws, "upper")
                x_bounds, y_lo = get_bound(x, y, ma_ws, "lower")
                plt.fill_between(x_bounds, y_lo, y_up, color=color, alpha=0.4, label=var_label)
        else:
            if x is not None:
                plt.plot(x, y, label=label, color=color)
            else:
                plt.plot(y, label=label, color=color)

    if mov_avg:
        return ma_ws * step_size
    else:
        return None


def add_empty_plot(label, color=None):
    plt.plot([], [], label=label, color=color)


def compare_statistics(model_names, env_type, labels=None,
                       cut_at_episode=None, cut_at_cycle=None, cut_at_transition=None, cut_at_hour=None,
                       plot_variance=False):
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

    if cut_at_episode is None:
        cut_at_episode = max_num_episodes
    if cut_at_cycle is None:
        cut_at_cycle = max_num_cycles

    out_path = "out/comparison_plots/"

    plot_everything(stats_collection=stats_collection,
                    labels=labels,
                    env_type=env_type,
                    out_path=out_path,
                    cut_at_episode=cut_at_episode,
                    cut_at_cycle=cut_at_cycle,
                    cut_at_transition=cut_at_transition,
                    cut_at_hour=cut_at_hour,
                    plot_variance=plot_variance)


def plot_everything(out_path, env_type: Environment, **kwargs):
    # (Re-)create output folder
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # Episode stats
    ep_plot_params = zip(["episodes", "transitions", "wall-clock_time"],
                         ["Episode", "Transition", "Wall-clock time (h)"])
    for domain, x_label in ep_plot_params:
        sub_out_path = out_path + domain + "/"
        os.mkdir(sub_out_path)
        kwargs.update({"domain": domain,
                       "x_label": x_label,
                       "out_path": sub_out_path})
        compare_returns(**kwargs)
        compare_scores(**kwargs)
        if env_type.TIME_RELEVANT:
            compare_times(**kwargs)
        if env_type.WINS_RELEVANT:
            compare_wins(**kwargs)

    # Cycle stats
    cyc_plot_params = zip(["cycles", "transitions", "wall-clock_time"],
                          ["Cycle", "Transition", "Wall-clock time (h)"])
    for domain, x_label in cyc_plot_params:
        sub_out_path = out_path + domain + "/"
        if not os.path.exists(sub_out_path):
            os.mkdir(sub_out_path)
        kwargs.update({"domain": domain,
                       "x_label": x_label,
                       "out_path": sub_out_path})
        compare_losses(**kwargs)
        compare_learning_rates(**kwargs)


def compare_returns(plot_variance, **kwargs):
    plot_comparison_on_domain(name="returns",
                              getter_handle=Statistics.get_returns,
                              orig_domain="episodes",
                              title="Return history comparison",
                              y_label="Return",
                              mov_avg=True,
                              logarithmic=False,
                              plot_variance=plot_variance,
                              **kwargs)
    plot_comparison_on_domain(name="return_records",
                              getter_handle=Statistics.get_return_records,
                              orig_domain="episodes",
                              title="Return records comparison",
                              y_label="Return",
                              mov_avg=False,
                              logarithmic=False,
                              plot_variance=False,
                              **kwargs)


def compare_scores(plot_variance, **kwargs):
    plot_comparison_on_domain(name="scores",
                              getter_handle=Statistics.get_scores,
                              orig_domain="episodes",
                              title="Score history comparison",
                              y_label="Score",
                              mov_avg=True,
                              logarithmic=False,
                              plot_variance=plot_variance,
                              **kwargs)
    plot_comparison_on_domain(name="score_records",
                              getter_handle=Statistics.get_score_records,
                              orig_domain="episodes",
                              title="Score records comparison",
                              y_label="Score",
                              mov_avg=False,
                              logarithmic=False,
                              plot_variance=False,
                              **kwargs)


def compare_times(plot_variance, **kwargs):
    plot_comparison_on_domain(name="times",
                              getter_handle=Statistics.get_times,
                              orig_domain="episodes",
                              title="Time history comparison",
                              y_label="Episode length",
                              mov_avg=True,
                              logarithmic=True,
                              plot_variance=plot_variance,
                              **kwargs)
    plot_comparison_on_domain(name="time_records",
                              getter_handle=Statistics.get_time_records,
                              orig_domain="episodes",
                              title="Time records comparison",
                              y_label="Episode length",
                              mov_avg=False,
                              logarithmic=True,
                              plot_variance=False,
                              **kwargs)


def compare_wins(**kwargs):
    plot_comparison_on_domain(name="wins",
                              getter_handle=Statistics.get_wins,
                              orig_domain="episodes",
                              title="Win-ratio history comparison",
                              y_label="Win proportion",
                              mov_avg=True,
                              logarithmic=False,
                              **kwargs)


def compare_losses(**kwargs):
    plot_comparison_on_domain(name="losses",
                              getter_handle=Statistics.get_losses,
                              orig_domain="cycles",
                              title="Loss history comparison",
                              y_label="Loss",
                              mov_avg=True,
                              logarithmic=True,
                              **kwargs)


def compare_learning_rates(**kwargs):
    kwargs.pop("plot_variance")
    plot_comparison_on_domain(name="learning-rates",
                              getter_handle=Statistics.get_learning_rates,
                              orig_domain="cycles",
                              title="Learning rate history comparison",
                              y_label="Learning rate",
                              mov_avg=False,
                              logarithmic=False,
                              plot_variance=False,
                              **kwargs)


def plot_comparison_on_domain(name, stats_collection, labels, getter_handle, domain, orig_domain, title,
                              mov_avg, x_label, y_label, out_path, logarithmic, plot_variance, **kwargs):
    if orig_domain == "episodes":
        get_domain = get_episodes_domain
    elif orig_domain == "cycles":
        get_domain = get_cycles_domain
    else:
        raise ValueError("Invalid original domain given.")

    x_lsts, y_lsts = [], []
    for stats in stats_collection:
        x = get_domain(stats, domain)
        y = getter_handle(stats)
        x, y = cut_data(x, y, domain, **kwargs)
        x_lsts += [x]
        y_lsts += [y]

    ma_ws = add_charts(x_lsts=x_lsts, y_lsts=y_lsts, labels=labels, mov_avg=mov_avg, zero2nan=logarithmic,
                       interp=domain != orig_domain, plot_confidence=plot_variance)
    if ma_ws is not None:
        y_label += " (%s MA)" % num2text(ma_ws)

    keep = logarithmic
    time_domain = domain == "wall-clock_time"
    finalize_plot(title=title, x_label=x_label, y_label=y_label, out_path=out_path + name + ".png",
                  legend=True, logarithmic=False, time_domain=time_domain, keep=keep)
    if logarithmic:
        finalize_plot(title=title, x_label=x_label, y_label=y_label, out_path=out_path + "log_" + name + ".png",
                      legend=True, logarithmic=True, time_domain=time_domain, keep=False)


def interpolate(x, y, step_size):
    if step_size == 0:
        return x, y
    x_eval = np.arange(start=x[0], stop=x[-1], step=step_size)
    y_interp = np.interp(x_eval, x, y, left=np.nan, right=np.nan)
    return x_eval, y_interp


def cut_data(x_values, y_values, domain,
             cut_at_episode, cut_at_cycle, cut_at_transition, cut_at_hour):
    if domain == "episodes" and cut_at_episode is not None:
        if x_values is not None:
            x_values = x_values[:cut_at_episode]
        y_values = y_values[:cut_at_episode]
    elif domain == "cycles" and cut_at_cycle is not None:
        if x_values is not None:
            x_values = x_values[:cut_at_cycle]
        y_values = y_values[:cut_at_cycle]
    elif domain == "transitions" and cut_at_transition is not None:
        x_values, y_values = cut_too_late(x_values, y_values, cut_at_transition)
    elif domain == "wall-clock_time" and cut_at_hour is not None:
        x_values, y_values = cut_too_late(x_values, y_values, cut_at_hour)
    return x_values, y_values


def cut_too_late(x_values, y_values, latest):
    too_late = x_values > latest
    if np.any(too_late):
        cut_episode = np.argmax(too_late)
        return x_values[:cut_episode], y_values[:cut_episode]
    else:
        return x_values, y_values


def get_episodes_domain(stats, domain):
    if domain == "episodes":
        return None
    elif domain == "transitions":
        return stats.get_episode_transitions()
    elif domain == "wall-clock_time":
        return stats.get_episode_hours()
    else:
        raise ValueError("Invalid domain name provided.")


def get_cycles_domain(stats, domain):
    if domain == "cycles":
        return None
    elif domain == "transitions":
        return stats.get_cycle_transitions()
    elif domain == "wall-clock_time":
        return stats.get_cycle_hours()
    else:
        raise ValueError("Invalid domain name provided.")


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
