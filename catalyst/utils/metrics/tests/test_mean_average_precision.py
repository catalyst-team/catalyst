# flake8: noqa
import math

import torch

from catalyst.utils.metrics import mean_average_precision


def test_mean_average_precision_weighted():
    """
    Tests for catalyst.utils.metrics.mean_average_precision metric.
    """
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0.1, 0.2, 0.3, 4])
    weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])

    ap = mean_average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 + 1.1 * 1 / 3.1 + 0 * 1.0 / 4.0) / 2.0

    assert math.fabs(ap[0] - val) < 0.01, "mAP test1 failed"

    ap = mean_average_precision(outputs=output, targets=target, weights=None)
    val = (
        1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 + 2.0 * 1.0 / 3.0 + 0 * 1.0 / 4.0
    ) / 2.0
    assert math.fabs(ap[0] - val) < 0.01, "mAP test2 failed"

    # Test multiple K's
    target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
    output = torch.Tensor([[0.1, 0.2, 0.3, 4], [4, 3, 2, 1]]).transpose(0, 1)
    weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
    ap = mean_average_precision(outputs=output, targets=target, weights=weight)

    assert (
        math.fabs(
            ap[0]
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
            ).mean()
        )
        < 0.01
    ), "mAP test3 failed"

    ap = mean_average_precision(outputs=output, targets=target, weights=None)
    assert (
        math.fabs(
            ap[0]
            - torch.Tensor(
                [
                    (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0 + 0 * 1.0 / 4.0)
                    / 2.0,
                    (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2 * 1.0 / 4.0)
                    / 2.0,
                ]
            ).mean()
        )
        < 0.01
    ), "mAP test4 failed"
