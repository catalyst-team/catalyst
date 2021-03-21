import torch

from catalyst.utils import metrics


def test_dice():
    """
    Tests for catalyst.utils.metrics.hitrate metric.
    """
    assert (
        metrics.hitrate(torch.tensor([[1], [2], [3]]), torch.tensor([2, 3, 4]))
        == 0
    )
    assert (
        metrics.hitrate(torch.tensor([[1], [2], [3]]), torch.tensor([1, 2, 3]))
        == 1
    )

    assert (
        metrics.hitrate(
            torch.tensor([[2, 0], [1, 3], [4, 6]]), torch.tensor([2, 3, 6])
        )
        == 1
    )
    assert (
        metrics.hitrate(
            torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            torch.tensor([1, 3, 1, 2]),
        )
        == 0.5
    )

    assert (
        metrics.hitrate(
            torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]]),
            torch.tensor([1, 3, 1, 2]),
        )
        == 0.5
    )

    assert (
        metrics.hitrate(
            torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]]),
            torch.tensor([1, 3, 1, 2]),
        )
        == 0.5
    )
