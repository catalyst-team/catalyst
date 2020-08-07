"""
In this module **precision**, **recall** and **F1 score**
calculations are defined in separate functions.

:py:class:`PrecisionRecallF1ScoreMeter` can keep track for all three of these.
"""
from collections import defaultdict

import torch

from catalyst.tools.meters import meter


def f1score(precision_value, recall_value, eps=1e-5):
    """
    Calculating F1-score from precision and recall to reduce computation
    redundancy.

    Args:
        precision_value: precision (0-1)
        recall_value: recall (0-1)
        eps: epsilon to use

    Returns:
        F1 score (0-1)
    """
    numerator = 2 * (precision_value * recall_value)
    denominator = precision_value + recall_value + eps
    return numerator / denominator


def precision(tp, fp, eps: float = 1e-5) -> float:
    """
    Calculates precision (a.k.a. positive predictive value) for binary
    classification and segmentation.

    Args:
        tp (int): number of true positives
        fp (int): number of false positives
        eps: epsilon to use

    Returns:
        precision value (0-1)
    """
    # originally precision is: ppv = tp / (tp + fp + eps)
    # but when both masks are empty this gives: tp=0 and fp=0 => ppv=0
    # so here precision is defined as ppv := 1 - fdr (false discovery rate)
    return 1 - fp / (tp + fp + eps)


def recall(tp, fn, eps=1e-5) -> float:
    """
    Calculates recall (a.k.a. true positive rate) for binary classification and
    segmentation.

    Args:
        tp: number of true positives
        fn: number of false negatives
        eps: epsilon to use

    Returns:
        recall value (0-1)
    """
    # originally reacall is: tpr := tp / (tp + fn + eps)
    # but when both masks are empty this gives: tp=0 and fn=0 => tpr=0
    # so here recall is defined as tpr := 1 - fnr (false negative rate)
    return 1 - fn / (fn + tp + eps)


class PrecisionRecallF1ScoreMeter(meter.Meter):
    """
    Keeps track of global true positives, false positives, and false negatives
    for each epoch and calculates precision, recall, and F1-score based on
    those metrics. Currently, this meter works for binary cases only, please
    use multiple instances of this class for multi-label cases.
    """

    def __init__(self, threshold=0.5):
        """
        Constructor method for the `` PrecisionRecallF1ScoreMeter`` class.
        """
        super(PrecisionRecallF1ScoreMeter, self).__init__()
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets true positive, false positive and false negative counts to 0.
        """
        self.tp_fp_fn_counts = defaultdict(int)

    def add(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """
        Thresholds predictions and calculates the true positives,
        false positives, and false negatives in comparison to the target.

        Args:
            output (torch.Tensor): prediction after activation function
                shape should be (batch_size, ...), but works with any shape
            target (torch.Tensor): label (binary),
                shape should be the same as output's shape
        """
        output = (output > self.threshold).float()

        tp = torch.sum(target * output)
        fp = torch.sum(output) - tp
        fn = torch.sum(target) - tp

        self.tp_fp_fn_counts["tp"] += tp
        self.tp_fp_fn_counts["fp"] += fp
        self.tp_fp_fn_counts["fn"] += fn

    def value(self):
        """
        Calculates precision/recall/f1 based on the current stored
        tp/fp/fn counts.

        Returns:
            tuple of floats: (precision, recall, f1)
        """
        precision_value = precision(
            self.tp_fp_fn_counts["tp"], self.tp_fp_fn_counts["fp"]
        )
        recall_value = recall(
            self.tp_fp_fn_counts["tp"], self.tp_fp_fn_counts["fn"]
        )
        f1_value = f1score(precision_value, recall_value)
        return (float(precision_value), float(recall_value), float(f1_value))


__all__ = ["PrecisionRecallF1ScoreMeter"]
