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
        window_aspect_ratio = self.screen_shape[1] / self.screen_shape[0]
        window_placements = spread_windows(self.num_screens, window_aspect_ratio)
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


def spread_windows(n, window_aspect_ratio=1):
    """Returns window coordinates and sizes for n windows distributed evenly over the entire screen.
    Window aspect ratio is width / height."""
    assert n > 0

    # Constants
    win_marg = 0.1  # space between windows, percent of window size
    vertical_screen_pad = 0.05  # padding tob and bottom each, percent of screen height
    horizontal_screen_pad = 0.05  # padding left and right each, percent of screen width
    max_rel_win_size = 0.5  # percent of screen size

    # Get screen info
    screen = screeninfo.get_monitors()[0]
    screen_aspect_ratio = screen.width / screen.height

    # Determine render area width and height
    render_area_width = screen.width * (1 - 2 * horizontal_screen_pad)
    render_area_height = screen.height * (1 - 2 * vertical_screen_pad)

    # Set maximum window width to avoid oversized windows
    max_win_width = int(min(screen.width * max_rel_win_size,
                            screen.height * max_rel_win_size * window_aspect_ratio))

    # Set initial number of columns and rows
    cols = n
    rows = 1

    # Set initial window width (same for all windows)
    w = render_area_width / (cols * (1 + win_marg))
    w = min(w, max_win_width)

    # Try out different row and column numbers to get largest possible window width
    for rows in range(2, n):
        cols = int(np.ceil(n / rows))

        # Predict window width and height (constant for all windows)
        w_exp = render_area_width / (cols + (cols - 1) * win_marg)
        h_exp = render_area_height / (rows + (rows - 1) * win_marg)
        w_exp = min(w_exp, max_win_width)
        w_exp = min(w_exp, h_exp * window_aspect_ratio)

        if w_exp > w:
            w = w_exp
        else:
            rows -= 1
            cols = int(np.ceil(n / rows))
            break

    # Compute window height
    h = w / window_aspect_ratio

    # Compute outer paddings and margins
    top_screen_pad = screen.height * vertical_screen_pad
    left_screen_pad = screen.width * horizontal_screen_pad
    top_area_marg = (render_area_height - rows * h - (rows - 1) * h * win_marg) / 2
    left_area_marg = (render_area_width - cols * w - (cols - 1) * w * win_marg) / 2

    # Compute window placement data
    window_placements = []
    for r in range(rows):
        for c in range(cols):
            x = int(c * w * (1 + win_marg) + left_screen_pad + left_area_marg)
            y = int(r * h * (1 + win_marg) + top_screen_pad + top_area_marg)
            window_placements += [(x, y, int(w), int(h))]

    return window_placements
