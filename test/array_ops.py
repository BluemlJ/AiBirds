import numpy as np
import pytest
import time

from src.utils.array_ops import piecewise_roll


@pytest.mark.parametrize("shift,piece_size", [
    (600, 100),
    (600, 10),
    (600, None),
    (100, 200),
    (100, None),
    (50, 200),
    (50, None),
    (800, 400),
    (800, None),
])
def test_piecewise_roll_is_correct(shift, piece_size):
    a = np.random.random((1000, 1000, 100, 2))

    # Original NumPy roll (ground truth)
    t = time.time()
    roll_1 = np.roll(a, shift=-shift, axis=0)
    roll_1[-shift:] = 0
    t_1 = time.time() - t

    # Piecewise roll
    t = time.time()
    roll_2 = a.copy()
    piecewise_roll(roll_2, shift=shift, piece_size=piece_size)
    t_2 = time.time() - t

    np.testing.assert_equal(roll_1, roll_2)
    print("NumPy roll took %.3f s and piecewise roll took %.3f s." % (t_1, t_2))
