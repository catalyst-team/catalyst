# flake8: noqa
import math

import numpy as np
import torch

from catalyst.metrics.functional.average_precision import (
    average_precision,
    binary_average_precision,
    mean_average_precision,
)


def test_binary_average_precision_base():
    """
    Tests for catalyst.binary_average_precision metric.
    """
    outputs = torch.Tensor([0.1, 0.4, 0.35, 0.8])
    targets = torch.Tensor([0, 0, 1, 1])

    assert torch.isclose(
        binary_average_precision(outputs, targets), torch.tensor(0.8333), atol=1e-3
    )


def test_binary_average_precision_weighted():
    """
    Tests for catalyst.binary_average_precision metric.
    """
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0.1, 0.2, 0.3, 4])
    weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 + 1.1 * 1 / 3.1 + 0 * 1 / 4) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test1 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test2 failed"

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([4, 3, 2, 1])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (0 * 1.0 / 1.0 + 1.0 * 2.0 / 3.0 + 2.0 * 0 / 6.0 + 6.0 * 1.0 / 10.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test3 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test4 failed"

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (4 * 1.0 / 4.0 + 6 * 1.0 / 6.0 + 0 * 6.0 / 9.0 + 0 * 6.0 / 10.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test5 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 2 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test6 failed"

    target = torch.Tensor([0, 0, 0, 0])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1.0, 0.1, 0.0, 0.5])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, "ap test7 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, "ap test8 failed"

    target = torch.Tensor([1, 1, 0])
    output = torch.Tensor([3, 1, 2])
    weight = torch.Tensor([1, 0.1, 3])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 1.0 / 1.0 + 1 * 0.0 / 4.0 + 1.1 / 4.1) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test9 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0) / 2.0
    assert math.fabs(ap - val) < 0.01, "ap test10 failed"

    # Test multiple K's
    target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
    output = torch.Tensor([[0.1, 0.2, 0.3, 4], [4, 3, 2, 1]]).transpose(0, 1)
    weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    assert (
        math.fabs(
            ap.sum()
            - torch.Tensor(
                [
                    (1 * 3.0 / 3.0 + 0 * 3.0 / 5.0 + 3.5 * 1 / 5.5 + 0 * 3.5 / 6.5) / 2.0,
                    (0 * 1.0 / 1.0 + 1 * 0.5 / 1.5 + 0 * 0.5 / 3.5 + 1 * 3.5 / 6.5) / 2.0,
                ]
            ).sum()
        )
        < 0.01
    ), "ap test11 failed"

    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    assert (
        math.fabs(
            ap.sum()
            - torch.Tensor(
                [
                    (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3 + 0 * 1.0 / 4.0) / 2.0,
                    (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2.0 * 1.0 / 4.0) / 2.0,
                ]
            ).sum()
        )
        < 0.01
    ), "ap test12 failed"


def test_avg_precision():
    """
    Tests for catalyst.avg_precision metric.
    """
    # # check everything is relevant
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [1.0, 1.0, 1.0, 1.0]

    ap_val = average_precision(torch.Tensor([y_pred]), torch.Tensor([y_true]))
    assert ap_val[0] == 1

    # # check is everything is relevant for 3 users
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [1.0, 1.0, 1.0, 1.0]

    ap_val = average_precision(
        torch.Tensor([y_pred, y_pred, y_pred]), torch.Tensor([y_true, y_true, y_true]),
    )
    assert torch.equal(ap_val, torch.ones(3))

    # # check everything is irrelevant
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [0.0, 0.0, 0.0, 0.0]

    ap_val = average_precision(torch.Tensor([y_pred]), torch.Tensor([y_true]))
    assert ap_val[0] == 0

    # # check is everything is irrelevant for 3 users
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [0.0, 0.0, 0.0, 0.0]

    ap_val = average_precision(
        torch.Tensor([y_pred, y_pred, y_pred]), torch.Tensor([y_true, y_true, y_true]),
    )
    assert torch.equal(ap_val, torch.zeros(3))

    # # check 4 test with k
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0.0, 1.0, 1.0, 1.0]
    y_true2 = [0.0, 1.0, 0.0, 0.0]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    ap_val = average_precision(y_pred_torch, y_true_torch)

    assert np.isclose(ap_val[0], 0.6389, atol=1e-3)
    assert np.isclose(ap_val[1], 0.333, atol=1e-3)

    # check 5
    # Stanford Introdcution to information retrieval primer
    y_pred1 = np.arange(9, -1, -1)
    y_true1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    y_pred2 = np.arange(9, -1, -1)
    y_true2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    ap_val = average_precision(y_pred_torch, y_true_torch)

    assert np.isclose(ap_val[0], 0.6222, atol=1e-3)
    assert np.isclose(ap_val[1], 0.4429, atol=1e-3)


def test_mean_avg_precision():
    """
    Tests for catalyst.mean_avg_precision metric.
    """
    # check 1
    # Stanford Introdcution to information retrieval primer
    y_pred1 = np.arange(9, -1, -1)
    y_true1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    y_pred2 = np.arange(9, -1, -1)
    y_true2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    top_k = [10]
    map_at10 = mean_average_precision(y_pred_torch, y_true_torch, top_k)[0]

    assert np.allclose(map_at10, 0.5325, atol=1e-3)
