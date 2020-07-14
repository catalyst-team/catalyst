import torch

from catalyst.utils import metrics

def test_mrr():
    """
    Tests for catalyst.utils.metrics.mrr metric.
    """

    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    