import numpy as np
import json
import matplotlib.pyplot as plt

from src.utils.mem import ReplayMemory

WINDOW_SIZES = (1000, 10000, 100000)


class Statistics:
    def __init__(self):
        self.final_returns = []
        self.final_scores = []
        self.final_times = []
        self.train_losses = []

    def denote_stats(self, final_return, final_score, final_time):
        self.final_returns += [final_return]
        self.final_scores += [final_score]
        self.final_times += [final_time]

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
        return_record = np.max(self.final_returns, initial=0)
        score_record = np.max(self.final_scores, initial=0)
        time_record = np.max(self.final_times, initial=0)

        return return_record, score_record, time_record

    def get_stats_dict(self):
        return {"returns": self.final_returns, "scores": self.final_scores,
                "times": self.final_times, "losses": self.train_losses}

    def set_stats(self, stats_dict):
        self.final_returns = stats_dict["returns"]
        self.final_scores = stats_dict["scores"]
        self.final_times = stats_dict["times"]
        self.train_losses = stats_dict["losses"]

    def save(self, out_path):
        file_path = out_path + "stats.txt"
        stats_dict = self.get_stats_dict()
        with open(file_path, 'w') as json_file:
            json.dump(stats_dict, json_file)

    def load(self, in_path):
        file_path = in_path + "stats.txt"
        with open(file_path) as json_file:
            try:
                stats_dict = json.load(json_file)
                self.set_stats(stats_dict)
            except Exception as e:
                print("Couldn't load statistics. Perhaps the stats.txt file doesn't contain"
                      "JSON-compatible information.")
                print(e)
            else:
                print("Successfully loaded statistics.")

    def loss_stagnates(self):
        if len(self.train_losses) >= 1000:
            return np.abs(np.average(self.train_losses[-1000:-500]) - np.average(self.train_losses[-500:])) < 0.0002
        else:
            return False

    def print_stats(self, par_step, total_par_steps, num_envs, comp_time, epsilon, records_file):
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
                     "\n   Time record:        %d" % time_record + \
                     "\n   Loss (50 MA):       %.4f" % avg_loss
        print(stats_text)
        records_file.write(stats_text + "\n")

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

    def log_extreme_losses(self, individual_losses, trans_ids, predictions, targets, memory, env, log_file):
        moving_avg_loss = self.get_moving_avg_loss(10)
        if moving_avg_loss > 0:
            extreme_loss = individual_losses > (moving_avg_loss * 1000)
            if np.any(extreme_loss) and log_file is not None:
                print("\033[93mExtreme loss encountered!\033[0m")
                extreme_loss_ids = np.where(extreme_loss)[0]
                for i, idx in zip(extreme_loss_ids, trans_ids[extreme_loss_ids]):
                    log_file.write("\nExtreme loss encountered in train cycle %d!" % self.get_train_cycle() +
                                   "\nExample loss: %.4f" % individual_losses[i] +
                                   "\n10 moving avg. loss: %.4f" % moving_avg_loss +
                                   "\nPredicted Q-values: " + str(predictions[i]) +
                                   "\nTarget Q-values: " + str(targets[i]))
                    log_file.write(memory.get_trans_text(idx, env) + "\n")


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
    plt.hlines(average, xmin=0, xmax=number_chunks-1, colors=['#009d81'], label="Total average")
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
    histogram = plt.hist(priorities, bins=100, range=range)
    plt.title("Transition priorities in experience set")
    plt.xlabel("Priority value")
    plt.ylabel("Number of transitions")
    if output_path is not None:
        plt.savefig(output_path, dpi=800)
    plt.show()
    return histogram


def compare_statistics(model_names, env_name):
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

    plot_comparison(out_path="out/comparison_plots/returns.png", comparison_values=returns, labels=model_names,
                    title="Return history comparison", ylabel="Loss", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/scores.png", comparison_values=scores, labels=model_names,
                    title="Score history comparison", ylabel="Score", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/times.png", comparison_values=times, labels=model_names,
                    title="Episode length history comparison", ylabel="Time (game ticks)", window_size=10000)
    plot_comparison(out_path="out/comparison_plots/losses.png", comparison_values=losses, labels=model_names,
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
