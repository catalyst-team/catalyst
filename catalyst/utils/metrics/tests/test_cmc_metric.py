import numpy as np
import pytest

import torch

from catalyst.contrib.dl.callbacks.cmc_callback import (  # noqa: F401
    CMCScoreCallback,
)
from catalyst.utils.metrics.cmc_score import _cmc_score_count

TEST_DATA = [
    # (distance_matrix, conformity_matrix, topk, expected_value)
    (torch.tensor([[1, 2], [2, 1]]), torch.tensor([[0, 1], [1, 0]]), 1, 0.0),
    (
        torch.tensor([[0, 0.5], [0.0, 0.5]]),
        torch.tensor([[0, 1], [1, 0]]),
        1,
        0.5,
    ),
    (
        torch.tensor([[0, 0.5], [0.0, 0.5]]),
        torch.tensor([[0, 1], [1, 0]]),
        2,
        1,
    ),
]


@pytest.mark.parametrize(
    "distance_matrix,conformity_matrix,topk,expected", TEST_DATA
)
def test_metric_count(distance_matrix, conformity_matrix, topk, expected):
    """Simple test"""
    out = _cmc_score_count(
        distances=distance_matrix,
        conformity_matrix=conformity_matrix,
        topk=topk,
    )
    assert np.isclose(out, expected)
