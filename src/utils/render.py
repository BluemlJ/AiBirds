import screeninfo
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ParallelScreen:
    """An auxiliary class for envs with displayable screen-like states.
    Supports parallel rendering of multiple active envs."""

    def __init__(self, num_screens, screen_shape):
        """
        :param num_screens: Usually the number of parallel environments
        :param screen_shape: (x, y) for grayscale and (x, y, 3) for RGB
        """
        self.num_screens = num_screens
        self.screen_shape = screen_shape
        self.screens = np.zeros(shape=(self.num_screens, *self.screen_shape), dtype="uint8")
        self.screen_rendering_initialized = False

    def update_screens(self, screens, scr_ids=None):
        """Updates the screens of all envs. Values are expected to be int
        in range 0 to 256."""
        if scr_ids is None:
            self.screens[:] = screens
        else:
            self.screens[scr_ids] = screens

    def _init_windows(self):
        window_placements = spread_windows(self.num_screens)
        for scr_id in range(self.num_screens):
            win_name = 'Env %d' % scr_id
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            x, y, w, h = window_placements[scr_id]
            cv2.moveWindow(win_name, x, y)
            cv2.resizeWindow(win_name, w, h)
        self.screen_rendering_initialized = True

    def render(self):
        if not self.screen_rendering_initialized:
            self._init_windows()
        for scr_id in range(self.num_screens):
            image = cv2.cvtColor(self.screens[scr_id], cv2.COLOR_RGB2BGR)
            cv2.imshow('Env %d' % scr_id, image)
            cv2.waitKey(1)

    def plot_all_screens(self):
        for scr_id in range(self.num_screens):
            self.plot_screen(scr_id)

    def plot_screen(self, scr_id):
        if len(self.screen_shape) == 2:  # grayscale image
            plt.imshow(self.screens[scr_id], cmap="binary", vmin=0, vmax=1)
        else:  # RGB image
            plt.imshow(self.screens[scr_id])
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title("Env %d" % scr_id)
        plt.show()
        plt.close()


def spread_windows(n):
    """Returns window coordinates and sizes for n windows distributed evenly over the entire screen.
    Assumes the windows to be quadratic"""
    assert n > 0

    margin = 0.05  # relative, percent of window size
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
