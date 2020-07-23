# flake8: noqa
from itertools import chain

import numpy as np
import pytest

import torch

from catalyst.utils.metrics.cmc_score import cmc_score_count

EPS = 1e-4

TEST_DATA_SIMPLE = (
    # (distance_matrix, conformity_matrix,
    #  topk, expected_value)
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
    (
        torch.tensor([[1, 0.5, 0.2], [2, 3, 4], [0.4, 3, 4]]),
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        2,
        1 / 3,
    ),
    (torch.randn((10, 10)), torch.ones((10, 10)), 1, 1),
)

TEST_DATA_LESS_SMALL = (
    (
        torch.rand((10, 10)) + torch.tril(torch.ones((10, 10))),
        torch.eye(10),
        i,
        i / 10,
    )
    for i in range(1, 10)
)

TEST_DATA_GREATER_SMALL = (
    (
        torch.rand((10, 10)) + torch.triu(torch.ones((10, 10)), diagonal=1),
        torch.eye(10),
        i,
        i / 10,
    )
    for i in range(1, 10)
)

TEST_DATA_LESS_BIG = (
    (
        torch.rand((100, 100)) + torch.tril(torch.ones((100, 100))),
        torch.eye(100),
        i,
        i / 100,
    )
    for i in range(1, 101, 10)
)


@pytest.mark.parametrize(
    "distance_matrix,conformity_matrix,topk,expected", TEST_DATA_SIMPLE
)
def test_metric_count(distance_matrix, conformity_matrix, topk, expected):
    """Simple test"""
    out = cmc_score_count(
        distances=distance_matrix,
        conformity_matrix=conformity_matrix,
        topk=topk,
    )
    assert np.isclose(out, expected)


@pytest.mark.parametrize(
    "distance_matrix,conformity_matrix,topk,expected",
    chain(TEST_DATA_LESS_SMALL, TEST_DATA_LESS_BIG),
)
def test_metric_less(distance_matrix, conformity_matrix, topk, expected):
    """Simple test"""
    out = cmc_score_count(
        distances=distance_matrix,
        conformity_matrix=conformity_matrix,
        topk=topk,
    )
    assert out - EPS <= expected


@pytest.mark.parametrize(
    "distance_matrix,conformity_matrix,topk,expected",
    chain(TEST_DATA_GREATER_SMALL),
)
def test_metric_greater(distance_matrix, conformity_matrix, topk, expected):
    """Simple test"""
    out = cmc_score_count(
        distances=distance_matrix,
        conformity_matrix=conformity_matrix,
        topk=topk,
    )
    assert out + EPS >= expected
