import torch

from catalyst.metrics.dice import dice


def test_dice():
    """
    Tests for catalyst.utils.metrics.dice metric.
    """
    size = 4
    half_size = size // 2
    shape = (1, 1, size, size)

    # check 0: one empty
    empty = torch.zeros(shape)
    full = torch.ones(shape)
    assert dice(empty, full).item() == 0

    # check 0: no overlap
    left = torch.ones(shape)
    left[:, :, :, half_size:] = 0
    right = torch.ones(shape)
    right[:, :, :, :half_size] = 0
    assert dice(left, right).item() == 0

    # check 1: both empty, both full, complete overlap
    assert dice(empty, empty) == 1
    assert dice(full, full) == 1
    assert dice(left, left) == 1
