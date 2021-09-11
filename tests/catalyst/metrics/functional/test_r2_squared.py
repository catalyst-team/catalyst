# flake8: noqa
import numpy as np
import torch

from catalyst.metrics.functional._r2_squared import r2_squared


def test_r2_squared():
    """
  Tests for catalyst.metrics.r2_squared metric.
  """
    y_true = torch.tensor([2.5, 0.0, 2, 8])
    y_pred = torch.tensor([3, -0.5, 2, 7])
    val = r2_squared(y_true, y_pred)
    assert torch.isclose(val, torch.Tensor([0.9486]))
