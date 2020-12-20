# flake8: noqa
from itertools import chain

import numpy as np
import pytest

import torch

from catalyst.metrics.cmc_score import (
    cmc_score,
    cmc_score_count,
    masked_cmc_score,
)

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


@pytest.mark.parametrize(
    (
        "query_embeddings",
        "gallery_embeddings",
        "conformity_matrix",
        "available_samples",
        "topk",
        "expected",
    ),
    (
        (
            torch.tensor(
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1],]
            ).float(),
            torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0],]).float(),
            torch.tensor(
                [
                    [True, False, False],
                    [True, False, False],
                    [False, True, True],
                    [False, True, True],
                ]
            ),
            torch.tensor(
                [
                    [False, True, True],
                    [True, True, True],
                    [True, False, True],
                    [True, True, True],
                ]
            ),
            1,
            0.75,
        ),
        (
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],]
            ).float(),
            torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 1],]).float(),
            torch.tensor(
                [
                    [False, False, True],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            ),
            torch.tensor(
                [
                    [True, True, True],
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ]
            ),
            1,
            0.25,
        ),
    ),
)
def test_masked_cmc_score(
    query_embeddings,
    gallery_embeddings,
    conformity_matrix,
    available_samples,
    topk,
    expected,
):
    score = masked_cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        conformity_matrix=conformity_matrix,
        available_samples=available_samples,
        topk=topk,
    )
    assert score == expected


@pytest.mark.parametrize(
    (
        "query_embeddings",
        "gallery_embeddings",
        "conformity_matrix",
        "available_samples",
        "topk",
    ),
    (
        (
            torch.rand(size=(query_size, 32)).float(),
            torch.rand(size=(gallery_size, 32)).float(),
            torch.randint(
                low=0, high=2, size=(query_size, gallery_size)
            ).bool(),
            torch.ones(size=(query_size, gallery_size)).bool(),
            k,
        )
        for query_size, gallery_size, k in zip(
            list(range(10, 20)), list(range(25, 35)), list(range(1, 11))
        )
    ),
)
def test_masked_score(
    query_embeddings,
    gallery_embeddings,
    conformity_matrix,
    available_samples,
    topk,
) -> None:
    """
    In this test we just check that masked_cmc_score is equal to cmc_score
    when all the samples are available for for scoring.
    """
    masked_score = masked_cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        conformity_matrix=conformity_matrix,
        available_samples=available_samples,
        topk=topk,
    )
    score = cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        conformity_matrix=conformity_matrix,
        topk=topk,
    )
    assert masked_score == score
