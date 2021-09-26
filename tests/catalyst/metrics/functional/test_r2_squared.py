# flake8: noqa
import numpy as np

import torch

from catalyst.metrics.functional._r2_squared import r2_squared


def test_r2_squared():
    """
    Tests for catalyst.metrics.r2_squared metric.
    """
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([2.5, 0.0, 2, 8])
    val = r2_squared(y_pred, y_true)
    assert torch.isclose(val, torch.Tensor([0.9486]))
