"""
Moving average meter calculates average for moving window of values.
"""
import math

import torch

from . import meter


class MovingAverageValueMeter(meter.Meter):
    """
    MovingAverageValueMeter stores mean and standard deviation
    for population of array that is handled like a queue during updates.
    Queue(window) is filled with zeros from the start by default.
    Meter updates are applied online, one value for each update.
    Meter values are moving average and moving standard deviation.
    """

    def __init__(self, windowsize):
        """
        Args:
            windowsize (int): size of window of values, which is continuous
                and ends on last updated element
        """
        super(MovingAverageValueMeter, self).__init__()
        self.windowsize = windowsize
        self.valuequeue = torch.Tensor(windowsize)
        self.reset()

    def reset(self) -> None:
        """
        Reset sum, number of elements, moving variance and zero out window.
        """
        self.sum = 0.0
        self.n = 0
        self.var = 0.0
        self.valuequeue.fill_(0)

    def add(self, value: float) -> None:
        """Adds observation sample.

        Args:
            value (float): scalar
        """
        queueid = self.n % self.windowsize
        oldvalue = self.valuequeue[queueid]
        self.sum += value - oldvalue
        self.var += value * value - oldvalue * oldvalue
        self.valuequeue[queueid] = value
        self.n += 1

    def value(self):
        """Return mean and standard deviation of window.

        Returns:
            tuple of floats: (window mean, window std)
        """
        n = min(self.n, self.windowsize)
        mean = self.sum / max(1, n)
        std = math.sqrt(max((self.var - n * mean * mean) / max(1, n - 1), 0))
        return mean, std
