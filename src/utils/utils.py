import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K


def get_moving_avg(list, n):
    """Computes the moving average with window size n of a list of numbers. The output list
    is padded at the beginning."""
    mov_avg = np.cumsum(list, dtype=float)
    mov_avg[n:] = mov_avg[n:] - mov_avg[:-n]
    mov_avg = mov_avg[n - 1:] / n
    mov_avg = np.insert(mov_avg, 0, int(n / 2) * [None])
    return mov_avg


def plot_moving_average(values, title, ylabel, output_path):
    add_moving_avg_plot(values, 100, 'silver')
    add_moving_avg_plot(values, 500, 'black')
    add_moving_avg_plot(values, 2000, '#009d81')

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(output_path, dpi=400)
    plt.show()


def plot_win_loss_ratio(list_of_wins):
    add_moving_avg_plot(list_of_wins, 100, 'silver')
    add_moving_avg_plot(list_of_wins, 500, 'black')
    add_moving_avg_plot(list_of_wins, 2000, '#009d81')

    plt.title("Win-loss ratio")
    plt.xlabel("Episodes")
    plt.ylabel("Percentage")
    plt.axis([None, None, 0, 1])
    plt.legend()
    plt.savefig("plots/win-loss-ratio.png", dpi=400)
    plt.show()


def add_moving_avg_plot(values, window_size, color):
    if len(values) > window_size:
        mov_avg_ret = get_moving_avg(values, window_size)
        plt.plot(mov_avg_ret, label="Moving average %d" % window_size, c=color)


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


def plot_state(state):
    # De-normalize state into image
    image = np.reshape(state, (124, 124, 3))

    # Plot it
    plt.imshow(image)
    plt.show()


def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def plot_priorities(priorities):
    length = len(priorities)
    plt.bar(range(length), priorities)
    plt.title("Transition priorities in experience set")
    plt.xlabel("Transition")
    plt.ylabel("Priority")
    plt.savefig("plots/priorities.png", dpi=800)
    plt.show()


'''def plot_saliency_map(state, model):
    # NOT FINISHED
    # De-normalize state into the image
    state = np.reshape(state, (124, 124, 3))
    image = (state * 255).astype(np.int)

    # Show the image
    # plt.imshow(image)
    # plt.show()

    # get the gradients of the last convolutional layer 
    last_conv = model.get_layer('conv2d_3')
    # TODO get the gradients

    # Global average pooling, returning the pooled gradients as well as the activation maps from the last conv layer
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv.output[0]])
    pooled_grads_value, conv_layer_output = iterate([image])

    # Multiplying the gradients and the activation maps to get importance into them 
    for i in range(512):
        conv_layer_output[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output, axis=-1)

    # apply reLU so only positive features are displayed
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x, y] = np.max(heatmap[x, y], 0)

    # normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.imshow(heatmap)

    # combine heatmap and input image to get the saliency map and plot it
    upsample = resize(heatmap, (124, 124), preserve_range=True)
    plt.imshow(image)
    plt.imshow(upsample, alpha=0.5)
    plt.show()'''
