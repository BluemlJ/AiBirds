import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import shutil
import json

WINDOW_SIZES = (1000, 10000, 50000)


class Statistics:
    def __init__(self):
        self.final_returns = []
        self.final_scores = []
        self.final_times = []
        self.train_losses = []

        self.return_record = -np.inf
        self.score_record = 0
        self.time_record = 0

    def denote_stats(self, final_return, final_score, final_time):
        new_return_record = self.return_record < final_return

        self.final_returns += [final_return]
        self.final_scores += [final_score]
        self.final_times += [final_time]

        self.return_record = np.max((self.return_record, final_return))
        self.score_record = np.max((self.score_record, final_score))
        self.time_record = np.max((self.time_record, final_time))

        return new_return_record

    def denote_loss(self, loss):
        self.train_losses += [loss]

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

    def get_train_losses(self):
        return np.array(self.train_losses)

    def get_moving_avg(self):
        if self.get_length() >= 10000:
            avg_return = np.average(self.final_returns[-10000:])
            avg_score = np.average(self.final_scores[-10000:])
            avg_time = np.average(self.final_times[-10000:])
        else:
            avg_return = np.nan
            avg_score = np.nan
            avg_time = np.nan

        avg_loss = self.get_moving_avg_loss(50)

        return avg_return, avg_score, avg_time, avg_loss

    def get_moving_avg_loss(self, window_size):
        if self.get_train_cycle() >= window_size:
            avg_loss = np.average(self.train_losses[-window_size:])
        else:
            avg_loss = np.nan

        return avg_loss

    def get_records(self):
        if self.return_record is np.nan:
            self.compute_records()
        return self.return_record, self.score_record, self.time_record

    def compute_records(self):
        self.return_record = np.max(self.final_returns, initial=-np.inf)
        self.score_record = np.max(self.final_scores, initial=0)
        self.time_record = np.max(self.final_times, initial=0)

    def get_stats_dict(self):
        return {"returns": self.final_returns, "scores": self.final_scores,
                "times": self.final_times, "losses": self.train_losses}

    def set_stats(self, stats_dict):
        self.final_returns = stats_dict["returns"]
        self.final_scores = stats_dict["scores"]
        self.final_times = stats_dict["times"]
        self.train_losses = stats_dict["losses"]

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
                print("Couldn't load statistics. Perhaps the stats.txt file doesn't contain "
                      "JSON-compatible information.")
                print(e)
            else:
                print("Successfully loaded statistics from '%s'." % in_path)
        self.compute_records()

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

    def print_stats(self, par_step, total_par_steps, num_envs, comp_time, epsilon, logger):
        avg_return, avg_score, avg_time, avg_loss = self.get_moving_avg()
        return_record, score_record, time_record = self.get_records()
        done_transitions = int(par_step * num_envs / 1000)

        stats_text = "\n\033[1mParallel step %d/%d\033[0m (%d s last 100)" % \
                     (par_step, total_par_steps, np.round(comp_time)) + \
                     "\n   Completed episodes: %d" % self.get_length() + \
                     "\n   Transitions done:   %d k" % done_transitions + \
                     "\n   Epsilon:            %.3f" % epsilon.get_value() + \
                     "\n   Score (10 k MA):    %.1f" % avg_score + \
                     "\n   Score record:       %.0f" % score_record + \
                     "\n   Time (10 k MA):     %.1f" % avg_time + \
                     "\n   Time record:        %.0f" % time_record + \
                     "\n   Loss (50 MA):       %.4f" % avg_loss
        print(stats_text)
        logger.log_step_statistics(stats_text + "\n")

    def plot_stats(self, out_path, memory):
        if self.get_length() >= 200:
            plot_priorities(memory.get_priorities(),  # useful to determine replay size
                            output_path=out_path + "priorities.png")
            if self.get_train_cycle() > 10:
                plot_moving_average(self.get_train_losses(),
                                    title="Training loss history",
                                    window_sizes=(10, 50, 200),
                                    ylabel="Loss", xlabel="Train cycle",
                                    output_path=out_path + "loss.png",
                                    logarithmic=True)
            plot_moving_average(self.get_final_times(),
                                title="Episode length history",
                                ylabel="Time (game ticks)",
                                output_path=out_path + "times.png")
            plot_moving_average(self.get_final_scores(),
                                title="Score history",
                                ylabel="Score",
                                output_path=out_path + "scores.png")
            plot_moving_average(self.get_final_returns(),
                                title="Return history",
                                ylabel="Return",
                                output_path=out_path + "returns.png")

    def log_extreme_losses(self, individual_losses, trans_ids, predictions, targets, memory, env, logger):
        moving_avg_loss = self.get_moving_avg_loss(10)
        if moving_avg_loss > 0:
            extreme_loss = individual_losses > (moving_avg_loss * 1000)
            if np.any(extreme_loss):
                print("\033[93mExtreme loss encountered!\033[0m")
                extreme_loss_ids = np.where(extreme_loss)[0]
                for i, idx in zip(extreme_loss_ids, trans_ids[extreme_loss_ids]):
                    logger.log_extreme_loss(self.get_train_cycle(), individual_losses[i], moving_avg_loss,
                                            str(predictions[i]), str(targets[i]),
                                            memory.get_trans_text(idx, env) + "\n")


class Epsilon:
    """A simple class providing basic functions to handle decaying epsilon greedy exploration."""

    def __init__(self, value, decay_mode="exp", decay_rate=1, minimum=0):
        """
        :param value: Initial value of epsilon
        :param decay_rate: Decrease/anneal factor for epsilon used when decay() gets invoked
        :param minimum: The value epsilon becomes in the limit through invoking decay()
        """
        if value < minimum:
            raise ValueError("You must provide a value for epsilon larger than the minimum.")

        if decay_mode not in ["exp", "lin"]:
            raise ValueError("Invalid decay mode provided. You gave %s, but only 'exp' and 'lin' are allowed." %
                             decay_mode)

        self.volatile_val = value - minimum
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


class Observations:
    """A finite and efficient ring buffer temporally holding observed transitions."""

    def __init__(self, buffer_size, num_envs, image_state_shape, numerical_state_shape):
        self.size = buffer_size

        self.image_states = np.zeros(np.append([buffer_size, num_envs], image_state_shape), dtype='bool')
        self.numerical_states = np.zeros(np.append([buffer_size, num_envs], numerical_state_shape), dtype='float32')
        self.actions = np.zeros((buffer_size, num_envs), dtype='int')
        self.score_gains = np.zeros((buffer_size, num_envs), dtype='int')
        self.rewards = np.zeros((buffer_size, num_envs), dtype='float32')
        self.times = np.zeros((buffer_size, num_envs), dtype='uint')

        self.buff_ptr = 0  # the pointer pointing at the current buffer position
        self.ep_beg_ptrs = np.zeros(num_envs, dtype='int')  # pointing at each episode's first transition

        self.curr_scores = np.zeros(num_envs, dtype='int')
        self.curr_returns = np.zeros(num_envs, dtype='float')

    def save_observations(self, states, actions, scores, rewards, times):
        image_states, numerical_states = states
        score_gains = scores - self.curr_scores

        self.image_states[self.buff_ptr] = image_states
        self.numerical_states[self.buff_ptr] = numerical_states
        self.actions[self.buff_ptr] = actions
        self.score_gains[self.buff_ptr] = score_gains
        self.rewards[self.buff_ptr] = rewards
        self.times[self.buff_ptr] = times

        self.curr_scores = scores
        self.curr_returns += rewards

        self.increment()

    def increment(self):
        self.buff_ptr = (self.buff_ptr + 1) % self.size
        max_len_episodes = self.ep_beg_ptrs == self.buff_ptr
        self.ep_beg_ptrs[max_len_episodes] += 1
        self.ep_beg_ptrs[max_len_episodes] %= self.size

    def get_observations(self, idx):
        ep_beg_ptr = self.ep_beg_ptrs[idx]

        if ep_beg_ptr < self.buff_ptr:
            obs_image_states = self.image_states[ep_beg_ptr:self.buff_ptr, idx]
            obs_numerical_states = self.numerical_states[ep_beg_ptr:self.buff_ptr, idx]
            obs_actions = self.actions[ep_beg_ptr:self.buff_ptr, idx]
            obs_score_gains = self.score_gains[ep_beg_ptr:self.buff_ptr, idx]
            obs_rewards = self.rewards[ep_beg_ptr:self.buff_ptr, idx]
            obs_times = self.times[ep_beg_ptr:self.buff_ptr, idx]
        else:
            # Create fancy index for episode entries
            trans_ids = (list(range(ep_beg_ptr, self.size)) + list(range(self.buff_ptr)), idx)
            obs_image_states = self.image_states[trans_ids]
            obs_numerical_states = self.numerical_states[trans_ids]
            obs_actions = self.actions[trans_ids]
            obs_score_gains = self.score_gains[trans_ids]
            obs_rewards = self.rewards[trans_ids]
            obs_times = self.times[trans_ids]

        obs_states = [obs_image_states, obs_numerical_states]

        return obs_states, obs_actions, obs_score_gains, obs_rewards, obs_times

    def get_performance(self, idx):
        obs_score = self.curr_scores[idx]
        obs_return = self.curr_returns[idx]

        return obs_score, obs_return

    def begin_new_episode_for(self, ids):
        self.ep_beg_ptrs[ids] = self.buff_ptr
        self.curr_scores[ids] = 0
        self.curr_returns[ids] = 0


class Logger:
    """Manages log files: creates, writes and closes them."""

    def __init__(self, out_path):
        self.step_stats_file = open(out_path + "step_stats.txt", "a", buffering=1)
        self.records_file = open(out_path + "records.txt", "a", buffering=1)
        self.extreme_loss_file = open(out_path + "extreme_losses.txt", "a", buffering=1)
        self.log_file = open(out_path + "log.txt", "a", buffering=1)

    def log_step_statistics(self, step_statistics):
        self.step_stats_file.write(step_statistics)

    def log_new_record(self, obs_return, transition):
        text = "\nNew return record achieved! The new best return is %.2f.\n" % obs_return + \
               "This is how the episode ended:" + transition + "\n"
        self.records_file.write(text)

    def log_extreme_loss(self, train_cycle, loss, ma_loss, pred_q, target_q, trans_text):
        text = "\nExtreme loss encountered in train cycle %d!" % train_cycle + \
               "\nExample loss: %.4f" % loss + \
               "\n10 moving avg. loss: %.4f" % ma_loss + \
               "\nPredicted Q-values: " + pred_q + \
               "\nTarget Q-values: " + target_q + \
               trans_text
        self.extreme_loss_file.write(text)

    def log(self, text):
        self.log_file.write(text)

    def close(self):
        self.step_stats_file.close()
        self.records_file.close()
        self.extreme_loss_file.close()
        self.log_file.close()


def get_moving_avg(list, n):
    """Computes the moving average with window size n of a list of numbers. The output list
    is padded at the beginning."""
    mov_avg = np.cumsum(list, dtype=float)
    mov_avg[n:] = mov_avg[n:] - mov_avg[:-n]
    mov_avg = mov_avg[n - 1:] / n
    mov_avg = np.insert(mov_avg, 0, int(n / 2) * [None])
    return mov_avg


def plot_moving_average(values, title, ylabel, output_path, window_sizes=WINDOW_SIZES, xlabel="Episode",
                        logarithmic=False, validation_values=None, validation_period=None):
    add_moving_avg_plot(values, window_sizes[0], 'silver')
    add_moving_avg_plot(values, window_sizes[1], 'black')
    add_moving_avg_plot(values, window_sizes[2], '#009d81')

    if validation_values is not None and validation_period is not None:
        add_validation_plot(validation_values, validation_period)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logarithmic:
        plt.yscale("log")
    plt.legend()
    plt.savefig(output_path, dpi=400)
    plt.show()


def add_validation_plot(validation_values, validation_period):
    n = len(validation_values)
    if n > 0:
        x = list(range(0, n * validation_period, validation_period))
        plt.plot(x, validation_values, label="Validation", c="orange")


def add_moving_avg_plot(values, window_size, color=None, label=None):
    if label is None:
        label = "Moving average %d" % window_size

    if len(values) > window_size:
        mov_avg_ret = get_moving_avg(values, window_size)
        plt.plot(mov_avg_ret, label=label, c=color)


def plot_validation(values, title, ylabel, output_path):
    number_chunks = int(len(values) / 10)
    chunked_scores = np.array_split(values, number_chunks)
    avg_scores = np.sum(chunked_scores, axis=1) / 10
    average = np.average(values)

    plt.bar(range(number_chunks), avg_scores, color='silver', label="Average per level type")
    plt.hlines(average, xmin=0, xmax=number_chunks - 1, colors=['#009d81'], label="Total average")
    plt.title(title)
    plt.xlabel("Level type")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(output_path, dpi=400)
    plt.show()


def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def plot_priorities(priorities, output_path=None, range=None):
    # hist, bins = np.histogram(priorities, bins=100)
    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    histogram = plt.hist(priorities, range=range)  # bins=logbins
    plt.title("Transition priorities in experience set")
    plt.xlabel("Priority value")
    plt.ylabel("Number of transitions")
    # plt.xscale("log")
    if output_path is not None:
        plt.savefig(output_path, dpi=800)
    plt.show()
    return histogram


def compare_statistics(model_names, env_name, labels=None):
    """Takes a list of model names, retrieves their statistics and plots them."""
    base_path = "out/%s/" % env_name
    returns = []
    scores = []
    times = []
    losses = []

    for model_name in model_names:
        in_path = base_path + model_name + "/"
        stats = Statistics()
        stats.load(in_path)

        returns += [stats.get_final_returns()]
        scores += [stats.get_final_scores()]
        times += [stats.get_final_times()]
        losses += [stats.get_train_losses()]

    if not os.path.exists("out/comparison_plots"):
        os.mkdir("out/comparison_plots")

    if labels is None:
        labels = model_names

    plot_comparison(out_path="out/comparison_plots/returns.png", comparison_values=returns, labels=labels,
                    title="Return history comparison", ylabel="Return", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/scores.png", comparison_values=scores, labels=labels,
                    title="Score history comparison", ylabel="Score", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/log_scores.png", comparison_values=scores, labels=labels,
                    title="Score history comparison", ylabel="Score", window_size=10000, logarithmic=True)
    plot_comparison(out_path="out/comparison_plots/times.png", comparison_values=times, labels=labels,
                    title="Episode length history comparison", ylabel="Time (game ticks)", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/log_times.png", comparison_values=times, labels=labels,
                    title="Episode length history comparison", ylabel="Time (game ticks)", window_size=10000,
                    logarithmic=True)
    plot_comparison(out_path="out/comparison_plots/losses.png", comparison_values=losses, labels=labels,
                    title="Training loss history comparison", ylabel="Loss", xlabel="Train cycle", logarithmic=True,
                    window_size=50)


def plot_comparison(out_path, comparison_values, labels, title, ylabel, window_size, xlabel="Episode",
                    logarithmic=False):
    for values, label in zip(comparison_values, labels):
        add_moving_avg_plot(values, window_size, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logarithmic:
        plt.yscale("log")
    plt.legend()
    plt.savefig(out_path, dpi=400)
    plt.show()


def check_for_existing_model(path):
    if os.path.exists(path):
        ans = input("There is already a model saved at '%s'. You can either override (delete) the existing\n"
                    "model or you can abort the program. Do you want to override the model? (y/n)\n" % path)
        if ans == "y":
            shutil.rmtree(path)
        else:
            raise Exception("User aborted program.")


def hyperparams_to_json(out_path, num_parallel_envs, use_dueling, use_double, learning_rate, latent_dim,
                        latent_a_dim, latent_v_dim, obs_buf_size, exp_buf_size):

    hyperparams_dict = {"num_parallel_envs": num_parallel_envs,
                        "use_dueling": use_dueling,
                        "use_double": use_double,
                        "learning_rate": learning_rate,
                        "latent_dim": latent_dim,
                        "latent_a_dim": latent_a_dim,
                        "latent_v_dim": latent_v_dim,
                        "obs_buf_size": obs_buf_size,
                        "exp_buf_size": exp_buf_size}

    with open(out_path + "hyperparams.json", 'w') as outfile:
        json.dump(hyperparams_dict, outfile)
