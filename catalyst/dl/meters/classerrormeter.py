import numbers

import numpy as np
import torch

from . import meter


class ClassErrorMeter(meter.Meter):
    def __init__(self, topk=[1], accuracy=False):
        super(ClassErrorMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = np.atleast_1d(target.cpu().squeeze().numpy())
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        if np.ndim(output) == 1:
            output = output[np.newaxis]
        else:
            assert np.ndim(output) == 2, \
                "wrong output size (1D or 2D expected)"
            assert np.ndim(target) == 1, \
                "target and output do not match"
        assert target.shape[0] == output.shape[0], \
            "target and output do not match"
        topk = self.topk
        maxk = int(topk[-1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        correct = pred == target[:, np.newaxis].repeat(pred.shape[1], 1)

        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                "invalid k (this k was not provided at construction time)"
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]
