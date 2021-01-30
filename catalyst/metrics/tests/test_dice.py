import torch

from catalyst.metrics.dice import dice


def test_dice():
    """
    Tests for catalyst.metrics.dice metric.
    """
    size = 4
    half_size = size // 2
    shape = (1, 1, size, size)

    # check 0: one empty
    empty = torch.zeros(shape)
    full = torch.ones(shape)
    assert dice(empty, full, class_dim=1).item() == 0

    # check 0: no overlap
    left = torch.ones(shape)
    left[:, :, :, half_size:] = 0
    right = torch.ones(shape)
    right[:, :, :, :half_size] = 0
    assert dice(left, right, class_dim=1).item() == 0

    # check 1: both empty, both full, complete overlap
    assert dice(empty, empty, class_dim=1).item() == 1
    assert dice(full, full, class_dim=1).item() == 1
    assert dice(left, left, class_dim=1).item() == 1

    # check 0.5: half overlap
    top_left = torch.zeros(shape)
    top_left[:, :, :half_size, :half_size] = 1
    assert torch.isclose(dice(top_left, left, class_dim=1), torch.Tensor([[0.66666]]))

    # check multiclass: 0, 0, 1, 1, 1, 0.5
    a = torch.cat([empty, left, empty, full, left, top_left], dim=1)
    b = torch.cat([full, right, empty, full, left, left], dim=1)
    ans = torch.Tensor([0, 0, 1, 1, 1, 0.66666])
    assert torch.allclose(dice(a, b, class_dim=1), ans)

    aaa = torch.cat([a, a, a], dim=0)
    bbb = torch.cat([b, b, b], dim=0)
    assert torch.allclose(dice(aaa, bbb, class_dim=1), ans)
