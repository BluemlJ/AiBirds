import numpy as np
import matplotlib.pyplot as plt


def get_moving_avg(list, n):
    """Computes the moving average with window size n of a list of numbers. The output list
    is padded at the beginning."""
    mov_avg = np.cumsum(list, dtype=float)
    mov_avg[n:] = mov_avg[n:] - mov_avg[:-n]
    mov_avg = mov_avg[n - 1:] / n
    mov_avg = np.insert(mov_avg, 0, int(n / 2) * [None])
    return mov_avg


def plot_scores(scores):
    # Window sizes for moving average
    w1 = 100
    w2 = 500
    w3 = 2000

    if len(scores) > w1:
        mov_avg_ret = get_moving_avg(scores, w1)
        plt.plot(mov_avg_ret, label="Moving average %d" % w1, c='silver')

    if len(scores) > w2:
        mov_avg_ret = get_moving_avg(scores, w2)
        plt.plot(mov_avg_ret, label="Moving average %d" % w2, c='black')

    if len(scores) > w3:
        mov_avg_ret = get_moving_avg(scores, w3)
        plt.plot(mov_avg_ret, label="Moving average %d" % w3, c='#009d81', linewidth=1.5)

    plt.title("Scores gathered so far")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_state(state):
    # De-normalize state into image
    state = np.reshape(state, (124, 124, 3))
    image = (state * 255).astype(np.int)

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
    plt.savefig("priorities.png", dpi=800)
    plt.show()
