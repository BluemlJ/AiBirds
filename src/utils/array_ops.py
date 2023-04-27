import numpy as np


def del_first(a: np.array, n: int) -> None:
    """Overwrites first n rows of array a with 0 and moves them to the end of a."""
    assert 0 <= n <= len(a)

    if n == 0:
        return
    elif n == len(a):
        a[:] = 0
    else:
        piecewise_roll(a, shift=n)


def piecewise_roll(a: np.array, shift: int, piece_size=None):
    """Performs a step-by-step in-place roll, but filling up with zeros. Equivalent to
    np.roll(a, shift=-shift, axis=0) but with less RAM usage and refilling with zeros."""
    assert shift > 0

    n = len(a)

    if piece_size is None:
        piece_size = n // 10
    pieces = int(np.ceil((n - shift) / piece_size))

    for p in range(0, pieces):
        left = p * piece_size
        right = min(left + piece_size, n - shift)
        a[left:right] = a[shift+left:shift+right]

    a[-shift:] = 0
