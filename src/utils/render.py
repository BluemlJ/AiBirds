import screeninfo
import numpy as np


def spread_windows(n):
    """Returns window coordinates and sizes for n windows distributed evenly over the entire screen.
    Assumes the windows to be quadratic"""
    assert n > 0

    margin = 0.1  # relative, percent of window size
    screen = screeninfo.get_monitors()[0]
    aspect_ratio = screen.width / screen.height
    max_win_size = screen.width // 2

    # Determine a pleasant number of columns and rows of windows
    cols = n
    rows = 1
    for rows in range(2, n):
        cols = np.ceil(n / rows)
        if rows * aspect_ratio > cols:
            rows -= 1
            cols = int(np.ceil(n / rows))
            break

    w = int(screen.width // (cols * (1 + 2 * margin)))
    w = min(w, max_win_size)
    h = w

    # Arrange windows
    top_screen_pad = (screen.height - rows * h * (1 + 2 * margin)) // 2
    window_placements = []
    for r in range(rows):
        for c in range(cols):
            x = int(c * w * (1 + 2 * margin) + margin * w)
            y = int(r * h * (1 + 2 * margin) + top_screen_pad)
            window_placements += [(x, y, w, h)]

    return window_placements
