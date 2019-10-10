import numbers

import numpy as np
import torch

from . import meter
from collections import defaultdict

class PrecisionRecallF1ScoreMeter(meter.Meter):
    """
    Keeps track of global true positives, false positives, and false negatives for each epoch
    and calculates precision, recall, and F1-score based on those metrics.
    Currently, for binary cases only (use multiple instances for multi-label).
    """
    def __init__(self, threshold=0.5):
        super(PrecisionRecallF1ScoreMeter, self).__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp_fp_fn_counts = defaultdict(int)

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            "wrong output size (1D expected)"
        assert np.ndim(target) == 1, \
            "wrong target size (1D expected)"
        assert output.shape[0] == target.shape[0], \
            "number of outputs and targets does not match"
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            "targets should be binary (0, 1)"

        output = (output > self.threshold).astype(np.int32)

        tp = np.sum(target * output)
        fp = np.sum(output) - tp
        fn = np.sum(target) - tp

        self.tp_fp_fn_counts["tp"] += tp
        self.tp_fp_fn_counts["fp"] += fp
        self.tp_fp_fn_counts["fn"] += fn

    def value(self):
        precision_value = float(precision(self.tp_fp_fn_counts["tp"], self.tp_fp_fn_counts["fp"]))
        recall_value = float(recall(self.tp_fp_fn_counts["tp"], self.tp_fp_fn_counts["fn"]))
        f1_value = float(f1score(precision_value, recall_value))
        return (precision_value, recall_value, f1_value)

def f1score(precision_value, recall_value, eps=1e-5):
    """
    Calculating F1-score from precision and recall to reduce computation redundancy.
    Args:
        precision_value: precision (0-1)
        recall_value: recall (0-1)
    Returns:
        F1 score (0-1)
    """
    return 2 * (precision_value * recall_value) / (precision_value + recall_value + eps)

def precision(tp, fp, eps=1e-5):
    """
    Calculates precision (a.k.a. positive predictive value) for binary classification.
    Args:
        tp: number of true positives
        fp: number of false positives
    Returns:
        precision value (0-1)
    """
    return tp / (tp + fp + eps)

def recall(tp, fn, eps=1e-5):
    """
    Calculates recall (a.k.a. true positive rate) for binary classification/segmentation
    Args:
        tp: number of true positives
        fn: number of false negatives
    Returns:
        recall value (0-1)
    """
    return tp / (tp + fn + eps)

__all__ = ["PrecisionRecallF1ScoreMeter"]
