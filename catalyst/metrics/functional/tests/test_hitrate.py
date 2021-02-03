import numpy as np
import torch

from catalyst.metrics.functional.hitrate import hitrate


def test_hitrate():
    """
    Tests for catalyst.metrics.hitrate metric.
    """
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    k = [1, 2]

    hitrate_at1, hitrate_at2 = hitrate(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert hitrate_at1 == 1.0
    assert hitrate_at2 == 0.5

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]
    k = [2]

    hitrate_at2 = hitrate(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)[0]
    assert hitrate_at2 == 0.0
