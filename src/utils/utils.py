import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
import ctypes  # for flashing window in taskbar under Windows


def plot(title, xlabel, ylabel, output_path, legend=False, logarithmic=False, show=False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logarithmic:
        plt.yscale("log")
    if legend:
        plt.legend()
    plt.savefig(output_path, dpi=400)
    if show:
        plt.show()
    else:
        plt.close()


def plot_validation(values, title, ylabel, output_path):
    number_chunks = int(len(values) / 10)
    chunked_scores = np.array_split(values, number_chunks)
    avg_scores = np.sum(chunked_scores, axis=1) / 10
    average = np.average(values)

    plt.bar(range(number_chunks), avg_scores, color='silver', label="Average per level type")
    plt.hlines(average, xmin=0, xmax=number_chunks - 1, colors=['#009d81'], label="Total average")
    plot(title, "Level type", ylabel, output_path, True, False)


def plot_highscores(highscores_ai, highscores_human, output_path=None):
    number_levels = len(highscores_ai)
    normalized_highscores_ai = highscores_ai / highscores_human
    norm_highscore_diff = normalized_highscores_ai - 1

    bar_colors = ['tab:green' if norm_level_score >= 0 else 'tab:red' for norm_level_score in norm_highscore_diff]
    plt.bar(range(number_levels), norm_highscore_diff, color=bar_colors, bottom=1)
    plt.hlines(1, xmin=-0.5, xmax=number_levels - 0.5, colors="black", label="Human performance")
    plt.title("Highscore Comparison AI vs. Human")
    plt.xlabel("Level")
    plt.ylabel("AI's highscore compared to human score")
    axes = plt.gca()
    axes.set_ylim([-0.1, 2.1])
    if output_path is not None:
        plt.savefig(output_path + "highscores.png", dpi=800)
    plt.show()


def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def check_for_existing_model(path):
    if os.path.exists(path):
        question = "There is already a model saved at '%s'. You can either override (delete) the existing\n" \
                   "model or you can abort the program. Do you want to override the model? (y/n)" % path
        if user_agrees_to(question):
            shutil.rmtree(path)
        else:
            print("No files changed. Shutting down program.")
            quit()


def config2text(config: dict):
    text = ""
    for entry in config.keys():
        text += ("\n" + entry + ": " + str(config[entry]))
    return text


def config2json(config, out_path):
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def json2config(in_path):
    with open(in_path) as infile:
        config = json.load(infile)
    return config


def convert_secs_to_hhmmss(s):
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)


def user_agrees_to(question):
    """Makes a yes/no query to the user. Returns True if user answered yes, False if no, and repeats if
    the user's question was invalid."""
    # let window flash in Windows
    ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)

    # ask question to user and handle answer
    while True:
        ans = input(question + "\n")
        if ans == "y":
            return True
        elif ans == "n":
            return False
        else:
            print("Invalid answer. Please type 'y' for 'Yes' or type 'n' for 'No.'")


# Colors and formatting for console text
def orange(text):
    return "\033[33m" + text + "\033[0m"


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def red(text):
    return "\033[91m" + text + "\033[0m"


def bold(text):
    return "\033[1m" + text + "\033[0m"


def del_first(lst, n):
    """Deletes in-place the first n elements from list and fills it with zeros."""
    if n == 0:
        return

    m = len(lst) - n
    assert m >= 0
    lst[:m] = lst[n:]
    lst[m:] = 0


def num2text(num):
    if num < 1000:
        return str(num)
    elif num < 1000000:
        num_rd = np.round(num / 1000)
        text = "%dK" % num_rd
        return text
    else:
        num_rd = np.round(num / 1000000)
        text = "%dM" % num_rd
        return text


def pad_data_with_zero_instances(data, pad_len):
    """Takes data (e.g., a 'too small' batch) and pads it with zero-value instances at the end."""
    data_shape_len = len(data.shape)
    padding_spec = ((0, pad_len),) + (data_shape_len - 1) * ((0, 0),)
    return np.pad(data, padding_spec)


def copy_model(model):
    model_class = type(model)
    return model_class(**model.get_config())


def print_info(text):
    print(yellow(text))


def print_warning(text):
    print(orange(text))


def print_error(text):
    print(red(text))
