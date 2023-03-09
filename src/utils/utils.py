import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
import pickle
import ctypes  # for flashing window in taskbar under Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
import inspect
import tensorflow as tf

from numpy.random import RandomState  # , SeedSequence, MT19937
from src.utils.text_sty import print_warning, yellow, print_info


def finalize_plot(title=None, x_label=None, y_label=None, out_path=None, legend=False, logarithmic=False,
                  time_domain=False, show=False, keep=False):
    if out_path is None and not show:
        print_info("Info: A plot is generated but without any effect (neither show nor save).")

    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if legend:
        plt.legend()
    if logarithmic:
        plt.yscale("log")
    if time_domain:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(hrs2hhmm))
    if out_path is not None:
        plt.savefig(out_path, dpi=400)
    if show:
        assert not keep
        plt.show()
    elif not keep:
        plt.close()


def plot_grayscale(image, **kwargs):
    plt.imshow(image, cmap="binary", vmin=0, vmax=255)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    finalize_plot(**kwargs)


def plot_validation(values, title, ylabel, output_path):
    number_chunks = int(len(values) / 10)
    chunked_scores = np.array_split(values, number_chunks)
    avg_scores = np.sum(chunked_scores, axis=1) / 10
    average = np.average(values)

    plt.bar(range(number_chunks), avg_scores, color='silver', label="Average per level type")
    plt.hlines(average, xmin=0, xmax=number_chunks - 1, colors=['#009d81'], label="Total average")
    finalize_plot(title, "Level type", ylabel, output_path, True, False)


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
    question = "There is already a model saved at '%s'. Override? (y/n)" % path
    if user_agrees_to(yellow(question)):
        remove_folder(path)
    else:
        print("No files changed. Shutting down program.")
        quit()


def remove_folder(path):
    # os.rmdir(path)
    shutil.rmtree(path, ignore_errors=True)


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
        print_warning("Unable to unpickle data from '%s'!" % in_path)
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


def del_first(lst, n):
    """Deletes in-place the first n elements from list and fills it with zeros."""
    if n == 0:
        return

    m = len(lst) - n
    assert m >= 0
    lst[:m] = lst[n:]
    lst[m:] = 0


def pad_data_with_zero_instances(data, pad_len):
    """Takes data (e.g., a 'too small' batch) and pads it with zero-value instances at the end."""
    data_shape_len = len(data.shape)
    padding_spec = ((0, pad_len),) + (data_shape_len - 1) * ((0, 0),)
    return np.pad(data, padding_spec)


def copy_object_with_config(model):
    model_class = type(model)
    return model_class(**model.get_config())


def set_seed(seed):
    np.random.seed(seed)  # TODO: apply new best practice
    tf.random.set_seed(seed)
    # RandomState(MT19937(SeedSequence(seed)))


def setup_hardware(use_gpu=True, gpu_memory_limit=None):
    # When on Windows, use TensorFlow version 2.10.1 max! Otherwise, GPU not usable.
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # make GPU visible
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide GPU

    if use_gpu and gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices('GPU')
        memory_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
        tf.config.experimental.set_virtual_device_configuration(gpus[0], memory_config)


def split_params(params, fn):
    fn_params = {}
    accepted_params = inspect.signature(fn).parameters.keys()
    for param in list(params.keys()):
        if param in accepted_params:
            fn_params[param] = params.pop(param)
    other_params = params
    return fn_params, other_params


def log_model_graph(model, input_shapes, log_dir="tensorboard/graphs/"):
    """Logs the TF computation graph for Tensorboard. Run Tensorboard with
    tensorboard --logdir tensorboard/graphs"""

    if os.path.exists(log_dir):
        remove_folder(log_dir)

    @tf.function
    def trace_fn(x):
        return model(x)

    writer = tf.summary.create_file_writer(log_dir)

    dummy_input = [tf.random.uniform((64,) + input_shape) for input_shape in input_shapes]
    tf.summary.trace_on(graph=True, profiler=True)
    trace_fn(dummy_input)
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


def shapes2arrays(shapes, dtypes=None, preceded_by=None):
    if preceded_by is not None:
        shapes = [(*preceded_by, *shape) for shape in shapes]
    if dtypes is None:
        dtypes = len(shapes) * ["float32"]
    elif type(dtypes) == str:
        dtypes = len(shapes) * [dtypes]
    return [np.zeros(shape=shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)]


def increase_last_dim(shapes, factor):
    assert factor > 0
    if factor == 1:
        return shapes
    else:
        return [(*shape[:-1], shape[-1] * factor) for shape in shapes]


def random_choice_along_last_axis(p):
    assert np.all(np.abs(np.sum(p, axis=-1)) - 1 < 1e-6)
    r = np.expand_dims(np.random.rand(*p.shape[:-1]), axis=-1)
    return (p.cumsum(axis=-1) > r).argmax(axis=-1)


def element_wise_index(lst: list, indices):
    """Takes a list of iterables which all have same first dimension and returns the list
    of results when indexing each iterable with indices."""
    selected = []
    for iterable in lst:
        selected += [iterable[indices]]
    return selected


def get_state_from_stack(stack, idx, state_shapes):
    state = []
    for comp, shape in zip(stack, state_shapes):
        depth = shape[-1]
        state += [comp[..., idx * depth:(idx + 1) * depth]]
    return state
