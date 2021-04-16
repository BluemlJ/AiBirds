import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
import pickle
import ctypes  # for flashing window in taskbar under Windows
import os
import tensorflow as tf

from numpy.random import RandomState  # , SeedSequence, MT19937


def plot(title, x_label, y_label, out_path, legend=False, logarithmic=False,
         time_domain=False, show=False, keep=False):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend()
    if logarithmic:
        plt.yscale("log")
    if time_domain:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(hrs2hhmm))
    plt.savefig(out_path, dpi=400)
    if show:
        assert not keep
        plt.show()
    elif not keep:
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


def angle2vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def ask_to_override_model(path):
    question = "There is already a model saved at '%s'. You can either override (delete) the existing\n" \
               "model or you can abort the program. Do you want to override the model? (y/n)" % path
    if user_agrees_to(question):
        remove_folder(path)
    else:
        print("No files changed. Shutting down program.")
        quit()


def remove_folder(path):
    shutil.rmtree(path)


def config2text(config: dict):
    text = ""
    for entry in config.keys():
        text += ("\n" + entry + ": " + str(config[entry]))
    return text


def config2json(config, out_path):
    """Allows only JSON-serializable data, e.g., native python objects."""
    with open(out_path, 'w') as json_file:
        json.dump(config, json_file)


def data2pickle(data, out_path):
    """Allows any type of data."""
    with open(out_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def json2config(in_path):
    with open(in_path) as infile:
        config = json.load(infile)
    return config


def pickle2data(in_path):
    try:
        with open(in_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            return data
    except Exception as e:
        print(red("Unable to unpickle data from '%s'!" % in_path))
        print(e)
    return None


def sec2hhmmss(s):
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)


def hrs2hhmm(h, pos=None):
    m = int((h % 1) * 60)
    h = h // 1
    return "%d:%02d" % (h, m)


def sec2hrs(s):
    return s / 3600


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


def copy_object_with_config(model):
    model_class = type(model)
    return model_class(**model.get_config())


def print_info(text):
    print(yellow(text))


def print_warning(text):
    print(orange(text))


def print_error(text):
    print(red(text))


def set_seed(seed):
    np.random.seed(seed)  # TODO: apply new best practice
    tf.random.set_seed(seed)
    # RandomState(MT19937(SeedSequence(seed)))


def setup_hardware(use_gpu=True, gpu_memory_limit=None):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # make GPU visible
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide GPU

    if use_gpu and gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices('GPU')
        memory_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
        tf.config.experimental.set_virtual_device_configuration(gpus[0], memory_config)


def log_model_graph(model, input_shape, log_dir="tensorboard/graphs/"):
    if os.path.exists(log_dir):
        remove_folder(log_dir)

    @tf.function
    def trace_fn(x):
        return model(x)

    writer = tf.summary.create_file_writer(log_dir)

    input_shape_2d, input_shape_1d = input_shape
    input_2d = tf.random.uniform((1,) + input_shape_2d)
    input_1d = tf.random.uniform((1,) + (input_shape_1d,))

    tf.summary.trace_on(graph=True, profiler=True)
    trace_fn([input_2d, input_1d])
    with writer.as_default():
        tf.summary.trace_export(
            name="DQN_model",
            step=0,
            profiler_outdir=log_dir)


def bool2num(arr):
    """Maps a boolean numpy array to a float array, where True -> 1, False -> -1."""
    return arr.astype(dtype="float32") * 2 - 1


def num2bool(arr):
    """Maps a float numpy array to a boolean array, where 1 -> True, else False."""
    return arr == 1
