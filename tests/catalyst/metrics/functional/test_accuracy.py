# flake8: noqa
from typing import Union

import numpy as np
import pytest
import torch

from catalyst.metrics.functional._accuracy import accuracy, multilabel_accuracy

BATCH_SIZE = 4
NUM_CLASSES = 10


def test_accuracy_top1():
    """
    Tests for catalyst.metrics.accuracy metric.
    """
    for i in range(NUM_CLASSES):
        outputs = torch.zeros((BATCH_SIZE, NUM_CLASSES))
        outputs[:, i] = 1
        targets = torch.ones((BATCH_SIZE, 1)) * i

        top1, top3, top5 = accuracy(outputs, targets, topk=(1, 3, 5))
        assert np.isclose(top1, 1)
        assert np.isclose(top3, 1)
        assert np.isclose(top5, 1)


def test_accuracy_top3():
    """
    Tests for catalyst.metrics.accuracy metric.
    """
    outputs = (
        torch.linspace(0, NUM_CLASSES - 1, steps=NUM_CLASSES)
        .repeat(1, BATCH_SIZE)
        .view(-1, NUM_CLASSES)
    )

    for i in range(NUM_CLASSES):
        targets = torch.ones((BATCH_SIZE, 1)) * i

        top1, top3, top5 = accuracy(outputs, targets, topk=(1, 3, 5))
        assert np.isclose(top1, 1 if i >= NUM_CLASSES - 1 else 0)
        assert np.isclose(top3, 1 if i >= NUM_CLASSES - 3 else 0)
        assert np.isclose(top5, 1 if i >= NUM_CLASSES - 5 else 0)


@pytest.mark.parametrize(
    "outputs,targets,threshold,true_value",
    (
        (torch.tensor([[0, 0.8], [0.75, 0.5]]), torch.tensor([[0, 1], [1, 1]]), 0.7, 0.75),
        (
            torch.tensor([[0.0, 0.1, 0.2], [0.4, 0.7, 0.0], [0.6, 0.9, 0], [0, 1.0, 0.77]]),
            torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            0.5,
            0.666667,
        ),
        (torch.tensor([[0.9, 0.9], [0.0, 0.0]]), torch.tensor([[1, 1], [1, 1]]), 0.6, 0.5),
        (
            torch.tensor([[0.7, 0.5], [0.5, 0.8]]),
            torch.tensor([[1, 0], [1, 1]]),
            torch.tensor([0.6, 0.7]),
            0.75,
        ),
    ),
)
def test_multilabel_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: Union[float, torch.Tensor],
    true_value: float,
):
    """
    Test multilabel accuracy with single and multiple thresholds

    Args:
        outputs: tensor of outputs
        targets: tensor of true answers
        threshold: thresholds for multilabel classification
        true_value: expected metric value
    """
    value = multilabel_accuracy(outputs=outputs, targets=targets, threshold=threshold).item()
    assert np.isclose(value, true_value)
