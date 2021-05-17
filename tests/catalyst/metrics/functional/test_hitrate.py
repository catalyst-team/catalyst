# flake8: noqa
import numpy as np
import torch

from catalyst.metrics.functional._hitrate import hitrate


def test_hitrate():
    """
    Tests for catalyst.metrics.hitrate metric.
    """
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 1.0]
    k = [1, 2, 3]

    hitrate_at1, hitrate_at2, hitrate_at3 = hitrate(
        torch.Tensor([y_pred]), torch.Tensor([y_true]), k
    )
    assert hitrate_at1 == 0.5
    assert hitrate_at2 == 0.5
    assert hitrate_at3 == 1.0

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]
    k = [2]

    hitrate_at2 = hitrate(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)[0]
    assert hitrate_at2 == 0.0

    # check batch case
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 0.0, 0.0]
    k = [1, 2, 3, 4]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    hitrate_at1, hitrate_at2, hitrate_at3, hitrate_at4 = hitrate(y_pred_torch, y_true_torch, k)

    assert hitrate_at1 == 0.0
    assert hitrate_at2 == 0.25
    assert hitrate_at3 == 0.25
    assert hitrate_at4 == 0.5
