from typing import Dict, Iterable, Tuple

import numpy as np
import pytest
import torch

from catalyst.metrics._cmc_score import CMCMetric, ReidCMCMetric
from catalyst.metrics._metric import AccumulationMetric

COMPLEX_OUTPUT_TYPE = Iterable[
    Tuple[Iterable[str], int, int, Iterable[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
]


@pytest.fixture()
def generate_batched_data() -> COMPLEX_OUTPUT_TYPE:
    """
    Generate batched data of different shapes for accumulation metric test.

    Returns:
        tuple of fields name for accumulation, number of batches, number of samples,
        batched data itself and true value for data accumulation without batches
    """
    batched_data = []
    for _ in range(10):
        num_fields = np.random.randint(low=1, high=20)
        fields_names = [f"field_{i}" for i in range(num_fields)]
        fields_shapes = {
            field_name: np.random.randint(low=1, high=5, size=np.random.randint(low=1, high=5))
            for field_name in fields_names
        }
        true_values = {field_name: None for field_name in fields_names}

        num_batches = np.random.randint(low=1, high=30)
        num_samples = 0
        batches = []

        for _ in range(num_batches):
            batch_size = np.random.randint(low=1, high=100)
            num_samples += batch_size
            batch_data = {}
            for field_name in fields_names:
                data = torch.randint(
                    low=10, high=1000, size=(batch_size, *fields_shapes[field_name]),
                )
                if true_values[field_name] is None:
                    true_values[field_name] = data
                else:
                    true_values[field_name] = torch.cat((true_values[field_name], data))
                batch_data[field_name] = data
            batches.append(batch_data)
        batched_data.append((fields_names, num_batches, num_samples, batches, true_values))
    return batched_data


def test_accumulation(generate_batched_data) -> None:  # noqa: WPS442
    """
    Check if AccumulationMetric accumulates all the data correctly along one loader
    """
    for (fields_names, num_batches, num_samples, batches, true_values) in generate_batched_data:
        metric = AccumulationMetric(accumulative_fields=fields_names)
        metric.reset(num_batches=num_batches, num_samples=num_samples)
        for batch in batches:
            metric.update(**batch)
        for field_name in true_values:
            assert (true_values[field_name] == metric.storage[field_name]).all()


def test_accumulation_reset(generate_batched_data):  # noqa: WPS442
    """Check if AccumulationMetric accumulates all the data correctly with multiple resets"""
    for (fields_names, num_batches, num_samples, batches, true_values) in generate_batched_data:
        metric = AccumulationMetric(accumulative_fields=fields_names)
        for _ in range(5):
            metric.reset(num_batches=num_batches, num_samples=num_samples)
            for batch in batches:
                metric.update(**batch)
            for field_name in true_values:
                assert (true_values[field_name] == metric.storage[field_name]).all()


def test_accumulation_dtype():
    """Check if AccumulationMetric accumulates all the data with correct types"""
    batch_size = 10
    batch = {
        "field_int": torch.randint(low=0, high=5, size=(batch_size, 5)),
        "field_bool": torch.randint(low=0, high=2, size=(batch_size, 10), dtype=torch.bool),
        "field_float32": torch.rand(size=(batch_size, 4), dtype=torch.float32),
    }
    metric = AccumulationMetric(accumulative_fields=list(batch.keys()))
    metric.reset(num_samples=batch_size, num_batches=1)
    metric.update(**batch)
    for key in batch:
        assert (batch[key] == metric.storage[key]).all()
        assert batch[key].dtype == metric.storage[key].dtype


def _test_score(
    metric: AccumulationMetric, batch: Dict[str, torch.Tensor], true_values: Dict[str, float],
) -> None:
    """Check if given metric works correctly"""
    metric.reset(num_batches=1, num_samples=len(batch["embeddings"]))
    metric.update(**batch)
    values = metric.compute_key_value()
    for key in true_values:
        assert key in values
        assert values[key] == true_values[key]


@pytest.mark.parametrize(
    "batch,topk,true_values",
    (
        (
            {
                "embeddings": torch.tensor(
                    [
                        [1, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ]
                ).float(),
                "labels": torch.tensor([0, 0, 1, 1, 0, 1, 1]),
                "is_query": torch.tensor([1, 1, 1, 1, 0, 0, 0]).bool(),
            },
            (1, 3),
            {"cmc01": 0.75, "cmc03": 1.0},
        ),
    ),
)
def test_cmc_score(
    batch: Dict[str, torch.Tensor], topk: Iterable[int], true_values: Dict[str, float]
) -> None:
    """Check if CMCMetric works correctly"""
    metric = CMCMetric(
        embeddings_key="embeddings", labels_key="labels", is_query_key="is_query", topk_args=topk,
    )
    _test_score(metric=metric, batch=batch, true_values=true_values)


@pytest.mark.parametrize(
    "batch,topk,true_values",
    (
        (
            {
                "embeddings": torch.tensor(
                    [
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ]
                ).float(),
                "pids": torch.Tensor([0, 0, 1, 1, 0, 1, 1]).long(),
                "cids": torch.Tensor([0, 1, 1, 2, 0, 1, 3]).long(),
                "is_query": torch.Tensor([1, 1, 1, 1, 0, 0, 0]).bool(),
            },
            (1, 3),
            {"cmc01": 0.75, "cmc03": 1},
        ),
    ),
)
def test_reid_cmc_score(
    batch: Dict[str, torch.Tensor], topk: Iterable[int], true_values: Dict[str, float]
) -> None:
    """Check if CMCMetric works correctly"""
    metric = ReidCMCMetric(
        embeddings_key="embeddings",
        pids_key="pids",
        cids_key="cids",
        is_query_key="is_query",
        topk_args=topk,
    )
    _test_score(metric=metric, batch=batch, true_values=true_values)
