from typing import Sequence


class Timer:
    """Used to stop the time of different categories."""
    def __init__(self, categories: Sequence[str]):
        self.times = {c: 0 for c in categories}

    def add_time(self, category: str, time: float):
        """ Saves the passed time for the given category
        :param category: The category name
        :param time: in seconds
        """
        self.times[category] += time

    def reset(self):
        """Sets all timer categories to 0s."""
        for c in list(self.times.keys()):
            self.times[c] = 0

    def print(self):
        print("Timer report:")
        for c in list(self.times.keys()):
            print("%s: %.2f s" % (c, self.times[c]))
