import numpy as np


def sigmoid(x):
    """Computes the sigmoid normed between (-1, +1). To get tanh just replace x with 2*x."""
    return 2 / (1 + np.exp(-x)) - 1


def get_2d_rotation_matrix(angle):
    """Returns the mathematical rotation matrix for the given angle
    :param angle: Angle in degrees (counter-clockwise).
    :return: NumPy array
    """
    angle_rad = np.deg2rad(angle)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                     [np.sin(angle_rad), np.cos(angle_rad)]])
