import numpy as np
from sty import Style, RgbFg, fg, ef, rs

"""Colors and formatting for console text"""


def red(text):
    return "\033[91m" + text + "\033[0m"


def orange(text):
    fg.orange = Style(RgbFg(255, 150, 50))
    return fg.orange + text + fg.rs


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def bold(text):
    return "\033[1m" + text + "\033[0m"


def it(text):  # italic
    return "\033[3m" + text + "\033[0m"


def ul(text):  # underline
    return "\033[4m" + text + "\033[0m"


def print_info(text):
    print(yellow(text))


def print_warning(text):
    print(orange(text))


def print_error(text):
    print(red(text))


def num2text(num):
    if num == 0:
        return "0"
    elif np.abs(num) < 1:
        return "%.2f" % num
    elif np.abs(num) < 10 and num % 1 != 0:
        return "%.1f" % num
    elif np.abs(num) < 1000:
        return "%.0f" % num
    elif np.abs(num) < 10000:
        thousands = num / 1000
        return "%.1fK" % thousands
    elif np.abs(num) < 1e6:
        thousands = num / 1000
        return "%.0fK" % thousands
    elif np.abs(num) < 1e7:
        millions = num / 1e6
        return "%.1fM" % millions
    else:
        millions = num / 1e6
        return "%.0fM" % millions