import torch

from catalyst.contrib.dl.callbacks.cmc_callback import (  # noqa: F401
    CMCScoreCallback,
)
from catalyst.dl import Runner
from catalyst.utils import set_global_seed
from catalyst.utils.metrics.cmc_score import _cmc_score_count


class MetricRunner(Runner):
    def _handle_batch(self, batch):
        output = self.model(batch["features"])
        self.output = {"embeddings": output}


class QueryGalleryDataset(torch.utils.data.Dataset):
    def __init__(self):
        set_global_seed(42)
        self.embeddings = torch.randn((10, 10))
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        self.labels = dist.sample((10, 1))

    def __getitem__(self, item):
        output_dict = {}
        output_dict["features"] = self.embeddings[item]
        output_dict["labels"] = self.labels[item]
        output_dict["query"] = item > 4
        return output_dict

    def __len__(self):
        return self.embeddings.shape[0]


def test_metric_simple_0():
    """Simple test with two examples"""
    distances = torch.tensor([[1, 2], [2, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 0.0
    assert out == expected


def test_metric_simple_05():
    """Simple test with two examples"""
    distances = torch.tensor([[0, 0.5], [0.0, 0.5]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 0.5
    assert out == expected


def test_metric_simple_1():
    """Simple test with two examples"""
    distances = torch.tensor([[1, 0.5], [0.5, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 1.0
    assert out == expected


def test_metric_simple_1_k_2():
    """Simple test with topk=2"""
    distances = torch.tensor([[1, 2], [2, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix, topk=2
    )
    expected = 1.0
    assert out == expected


def test_pipeline():
    net = torch.nn.Linear(10, 4)
    dataset = QueryGalleryDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    runner = MetricRunner()
    runner.train(
        model=net,
        loaders={"valid": dataloader},
        main_metric="cmc_1",
        callbacks={"cmc": CMCScoreCallback(topk_args=(1, 2))},
        check=True,
    )
