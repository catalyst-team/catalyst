import numpy as np

import torch

from catalyst.utils import metrics

BATCH_SIZE = 4
NUM_CLASSES = 10


def test_accuracy_top1():
    """
    Tests for catalyst.utils.metrics.accuracy metric.
    """
    for i in range(NUM_CLASSES):
        outputs = torch.zeros((BATCH_SIZE, NUM_CLASSES))
        outputs[:, i] = 1
        targets = torch.ones((BATCH_SIZE, 1)) * i

        top1, top3, top5 = metrics.accuracy(outputs, targets, topk=(1, 3, 5))
        assert np.isclose(top1, 1)
        assert np.isclose(top3, 1)
        assert np.isclose(top5, 1)


def test_accuracy_top3():
    """
    Tests for catalyst.utils.metrics.accuracy metric.
    """
    outputs = (
        torch.linspace(0, NUM_CLASSES - 1, steps=NUM_CLASSES)
        .repeat(1, BATCH_SIZE)
        .view(-1, NUM_CLASSES)
    )

    for i in range(NUM_CLASSES):
        targets = torch.ones((BATCH_SIZE, 1)) * i

        top1, top3, top5 = metrics.accuracy(outputs, targets, topk=(1, 3, 5))
        assert np.isclose(top1, 1 if i >= NUM_CLASSES - 1 else 0)
        assert np.isclose(top3, 1 if i >= NUM_CLASSES - 3 else 0)
        assert np.isclose(top5, 1 if i >= NUM_CLASSES - 5 else 0)
