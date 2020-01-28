import math

import torch

from . import meter


class MovingAverageValueMeter(meter.Meter):
    def __init__(self, windowsize):
        super(MovingAverageValueMeter, self).__init__()
        self.windowsize = windowsize
        self.valuequeue = torch.Tensor(windowsize)
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0
        self.var = 0.0
        self.valuequeue.fill_(0)

    def add(self, value):
        queueid = (self.n % self.windowsize)
        oldvalue = self.valuequeue[queueid]
        self.sum += value - oldvalue
        self.var += value * value - oldvalue * oldvalue
        self.valuequeue[queueid] = value
        self.n += 1

    def value(self):
        n = min(self.n, self.windowsize)
        mean = self.sum / max(1, n)
        std = math.sqrt(max((self.var - n * mean * mean) / max(1, n - 1), 0))
        return mean, std
