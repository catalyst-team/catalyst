import numpy as np
import pytest

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl


class DummyModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(
            in_features=self.output_dim, out_features=self.output_dim
        )

    def forward(self, x):
        dummy_x = torch.randn(size=(x.shape[0], self.output_dim))
        return self.linear(dummy_x)


class NamedTensorDataset(TensorDataset):
    def __getitem__(self, item):
        return {
            "features": self.tensors[0][item],
            "targets": self.tensors[1][item],
        }


@pytest.fixture
def generate_no_kv_datasets():
    datasets = []
    for _ in range(10):
        # generate dataset's len and class numbers
        n_samples = np.random.randint(low=50, high=500)
        n_classes = np.random.randint(low=2, high=10)

        # generate dimension of features
        ndim = np.random.randint(low=1, high=5)
        feature_dims = np.random.randint(low=4, high=20, size=(ndim,))

        X = torch.rand(size=(n_samples, *feature_dims))
        y = torch.randint(low=0, high=n_classes, size=(n_samples,)).long()
        one_hot = torch.nn.functional.one_hot(y).float()

        dataset = TensorDataset(X, one_hot)
        datasets.append((dataset, n_classes))

    return datasets


@pytest.fixture
def generate_kv_datasets():
    datasets = []
    for _ in range(10):
        # generate dataset's len and class numbers
        n_samples = np.random.randint(low=50, high=500)
        n_classes = np.random.randint(low=2, high=10)

        # generate dimension of features
        ndim = np.random.randint(low=1, high=5)
        feature_dims = np.random.randint(low=4, high=20, size=(ndim,))

        X = torch.rand(size=(n_samples, *feature_dims))
        y = torch.randint(low=0, high=n_classes, size=(n_samples,)).long()
        one_hot = torch.nn.functional.one_hot(y).float()

        dataset = NamedTensorDataset(X, one_hot)
        datasets.append((dataset, n_classes))

    return datasets


def test_accumulation_no_kv(generate_no_kv_datasets) -> None:
    """
    Check if all the data was accumulated correctly in LoaderMetricCallback.
    """
    for dataset, n_classes in generate_no_kv_datasets:
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=32)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = DummyModel(output_dim=n_classes)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters())

        # we are not going to check metric's value, only accumulated data
        callback = dl.LoaderMetricCallback(
            prefix="metric", metric_fn=lambda *x: 1.0
        )

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            # we need run >1 epochs to check callback's on_loader_end and
            # on_loader_start methods
            num_epochs=2,
            callbacks=[dl.ControlFlowCallback(callback, loaders="valid")],
        )
        assert (
            dataset.tensors[1] == torch.from_numpy(callback.input["_data"])
        ).all()


def test_accumulation_kv(generate_kv_datasets) -> None:
    """
    Check if all the kv-data was accumulated correctly in LoaderMetricCallback.
    """
    for dataset, n_classes in generate_kv_datasets:
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=32)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = DummyModel(output_dim=n_classes)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters())

        # we are not going to check metric's value, only accumulated data
        callback = dl.LoaderMetricCallback(
            input_key=["features", "targets"],
            output_key=["logits"],
            prefix="metric",
            metric_fn=lambda **x: 1.0,
        )

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            # we need run >1 epochs to check callback's on_loader_end and
            # on_loader_start methods
            num_epochs=2,
            callbacks=[dl.ControlFlowCallback(callback, loaders="valid")],
        )

        for key in ["features", "targets"]:
            assert key in callback.input
        assert "logits" in callback.output

        expected_features = dataset.tensors[0]
        accumulated_features = torch.from_numpy(
            callback.input["features"]
        ).float()
        assert torch.allclose(expected_features, accumulated_features)

        expected_targets = dataset.tensors[1]
        accumulated_targets = torch.from_numpy(callback.input["targets"])
        assert (expected_targets == accumulated_targets).all()
