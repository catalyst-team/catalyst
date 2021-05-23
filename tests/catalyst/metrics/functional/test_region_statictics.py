# flake8: noqa
import torch

from catalyst.metrics.functional._segmentation import get_segmentation_statistics


def test_segmentation_statistics():
    size = 4
    half_size = size // 2
    shape = (1, 1, size, size)

    # check 0: one empty
    empty = torch.zeros(shape)
    full = torch.ones(shape)
    tp, fp, fn = get_segmentation_statistics(empty, full, class_dim=1)
    assert tp == torch.tensor([0.0]) and fp == torch.tensor([0.0]) and fn == torch.tensor([16.0])

    # check 0: no overlap
    left = torch.ones(shape)
    left[:, :, :, half_size:] = 0
    right = torch.ones(shape)
    right[:, :, :, :half_size] = 0
    tp, fp, fn = get_segmentation_statistics(left, right, class_dim=1)
    assert tp == torch.tensor([0.0]) and fp == torch.tensor([8.0]) and fn == torch.tensor([8.0])

    # check 1: both empty, both full, complete overlap
    tp, fp, fn = get_segmentation_statistics(empty, empty, class_dim=1)
    assert tp == torch.tensor([0.0]) and fp == torch.tensor([0.0]) and fn == torch.tensor([0.0])
    tp, fp, fn = get_segmentation_statistics(full, full, class_dim=1)
    assert tp == torch.tensor([16.0]) and fp == torch.tensor([0.0]) and fn == torch.tensor([0.0])
    tp, fp, fn = get_segmentation_statistics(left, left, class_dim=1)
    assert tp == torch.tensor([8.0]) and fp == torch.tensor([0.0]) and fn == torch.tensor([0.0])

    # check 0.5: half overlap
    top_left = torch.zeros(shape)
    top_left[:, :, :half_size, :half_size] = 1
    tp, fp, fn = get_segmentation_statistics(left, top_left, class_dim=1)
    assert tp == torch.tensor([4.0]) and fp == torch.tensor([4.0]) and fn == torch.tensor([0.0])

    # check multiclass
    a = torch.cat([empty, left, empty, full, left, top_left], dim=1)
    b = torch.cat([full, right, empty, full, left, left], dim=1)
    true_tp = torch.tensor([0.0, 0.0, 0.0, 16.0, 8.0, 4.0])
    true_fp = torch.tensor([0.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    true_fn = torch.tensor([16.0, 8.0, 0.0, 0.0, 0.0, 4.0])
    tp, fp, fn = get_segmentation_statistics(a, b, class_dim=1)
    assert torch.allclose(tp, true_tp)
    assert torch.allclose(fp, true_fp)
    assert torch.allclose(fn, true_fn)

    aaa = torch.cat([a, a, a], dim=0)
    bbb = torch.cat([b, b, b], dim=0)
    true_tp = torch.tensor([0.0, 0.0, 0.0, 48.0, 24.0, 12.0])
    true_fp = torch.tensor([0.0, 24.0, 0.0, 0.0, 0.0, 0.0])
    true_fn = torch.tensor([48.0, 24.0, 0.0, 0.0, 0.0, 12.0])
    tp, fp, fn = get_segmentation_statistics(aaa, bbb, class_dim=1)
    assert torch.allclose(tp, true_tp)
    assert torch.allclose(fp, true_fp)
    assert torch.allclose(fn, true_fn)
