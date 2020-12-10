import torch

from catalyst.metrics.iou import iou


def test_iou():
    """
    Tests for catalyst.metrics.iou metric.
    """
    size = 4
    half_size = size // 2
    shape = (1, 1, size, size)

    # check 0: one empty
    empty = torch.zeros(shape)
    full = torch.ones(shape)
    assert iou(empty, full).item() == 0

    # check 0: no overlap
    left = torch.ones(shape)
    left[:, :, :, half_size:] = 0
    right = torch.ones(shape)
    right[:, :, :, :half_size] = 0
    assert iou(left, right).item() == 0

    # check 1: both empty, both full, complete overlap
    assert iou(empty, empty) == 1
    assert iou(full, full) == 1
    assert iou(left, left) == 1

    # check 0.5: half overlap
    top_left = torch.zeros(shape)
    top_left[:, :, :half_size, :half_size] = 1
    assert iou(top_left, left).item() == 0.5

    # @TODO: restore
    # check multiclass: 0, 0, 1, 1, 1, 0.5
    # a = torch.cat([empty, left, empty, full, left, top_left], dim=1)
    # b = torch.cat([full, right, empty, full, left, left], dim=1)
    # ans = torch.Tensor([0, 0, 1, 1, 1, 0.5])
    # assert torch.all(iou(a, b, classes=["dummy"]) == ans)
