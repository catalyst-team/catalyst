# flake8: noqa
import math

import torch

from catalyst.utils.metrics import average_precision


def test_average_precision_base():
    """
    Tests for catalyst.utils.metrics.average_precision metric.
    """
    outputs = torch.Tensor([0.1, 0.4, 0.35, 0.8])
    targets = torch.Tensor([0, 0, 1, 1])

    assert torch.isclose(
        average_precision(outputs, targets), torch.tensor(0.8333), atol=1e-3
    )


def test_average_precision_weighted():
    """
    Tests for catalyst.utils.metrics.average_precision metric.
    """
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0.1, 0.2, 0.3, 4])
    weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
    ap = average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 + 1.1 * 1 / 3.1 + 0 * 1 / 4) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test1 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test2 failed"

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([4, 3, 2, 1])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = average_precision(outputs=output, targets=target, weights=weight)
    val = (
        0 * 1.0 / 1.0 + 1.0 * 2.0 / 3.0 + 2.0 * 0 / 6.0 + 6.0 * 1.0 / 10.0
    ) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test3 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    val = (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test4 failed"

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = average_precision(outputs=output, targets=target, weights=weight)
    val = (
        4 * 1.0 / 4.0 + 6 * 1.0 / 6.0 + 0 * 6.0 / 9.0 + 0 * 6.0 / 10.0
    ) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test5 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 2 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test6 failed"

    target = torch.Tensor([0, 0, 0, 0])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1.0, 0.1, 0.0, 0.5])
    ap = average_precision(outputs=output, targets=target, weights=weight)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, "ap test7 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, "ap test8 failed"

    target = torch.Tensor([1, 1, 0])
    output = torch.Tensor([3, 1, 2])
    weight = torch.Tensor([1, 0.1, 3])
    ap = average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 1.0 / 1.0 + 1 * 0.0 / 4.0 + 1.1 / 4.1) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test9 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test10 failed"

    # Test multiple K's
    target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
    output = torch.Tensor([[0.1, 0.2, 0.3, 4], [4, 3, 2, 1]]).transpose(0, 1)
    weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
    ap = average_precision(outputs=output, targets=target, weights=weight)
    assert (
        math.fabs(
            ap.sum()
            - torch.Tensor(
                [
                    (
                        1 * 3.0 / 3.0
                        + 0 * 3.0 / 5.0
                        + 3.5 * 1 / 5.5
                        + 0 * 3.5 / 6.5
                    )
                    / 2.0,
                    (
                        0 * 1.0 / 1.0
                        + 1 * 0.5 / 1.5
                        + 0 * 0.5 / 3.5
                        + 1 * 3.5 / 6.5
                    )
                    / 2.0,
                ]
            ).sum()
        )
        < 0.01
    ), "ap test11 failed"

    ap = average_precision(outputs=output, targets=target, weights=None)
    assert (
        math.fabs(
            ap.sum()
            - torch.Tensor(
                [
                    (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3 + 0 * 1.0 / 4.0)
                    / 2.0,
                    (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2.0 * 1.0 / 4.0)
                    / 2.0,
                ]
            ).sum()
        )
        < 0.01
    ), "ap test12 failed"
