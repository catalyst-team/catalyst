import torch

from catalyst.utils import metrics


def test_hitrate():
    """
    Tests for catalyst.utils.metrics.hitrate metric.
    """
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    k = 2

    hitrate_at_2 = metrics.hitrate_at_k(
        torch.Tensor([y_pred]),
        torch.Tensor([y_true]),
        k
    )
    assert hitrate_at_2 == 0.5

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]
    k = 2

    hitrate_at_2 = metrics.hitrate_at_k(
        torch.Tensor([y_pred]),
        torch.Tensor([y_true]),
        k
    )
    assert hitrate_at_2 == 0.0
