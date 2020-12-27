from typing import Dict, List

import numpy as np
import pytest

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor


class DummyModel(nn.Module):
    """Model that generates random output"""

    def __init__(self, output_dim: int) -> None:
        """
        Args:
            output_dim: dim of model output
        """
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(
            in_features=self.output_dim, out_features=self.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input batch

        Returns:
            random output of size (batch_size, self.output_dim)
        """
        dummy_x = torch.randn(size=(x.shape[0], self.output_dim))
        return self.linear(dummy_x)


class NamedTensorDataset(TensorDataset):
    """
    TensorDataset wrapper that returns key-value items.
    """

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Args:
            item: index of item to return

        Returns:
            dict of dataset items
        """
        return {
            "features": self.tensors[0][item],
            "targets": self.tensors[1][item],
            "bool_field": self.tensors[2][item],
            "int_field": self.tensors[3][item],
        }


@pytest.fixture
def generate_no_kv_datasets() -> List[TensorDataset]:
    """
    Generate 10 TensorDatasets with different sizes, numbers of labels,
    feature dims.

    Returns:
        list of datasets
    """
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
def generate_kv_datasets() -> List[TensorDataset]:
    """
    Generate 10 NamedTensorDataset with different sizes, numbers of labels,
    feature dims and data types.

    Returns:
        list of datasets
    """
    datasets = []
    for _ in range(10):
        # generate dataset's len and class numbers
        n_samples = np.random.randint(low=50, high=500)
        n_classes = np.random.randint(low=2, high=10)

        # generate dimension of features
        ndim = np.random.randint(low=1, high=5)
        feature_dims = np.random.randint(low=4, high=20, size=(ndim,))

        X = torch.rand(size=(n_samples, *feature_dims)).float()
        y = torch.randint(low=0, high=n_classes, size=(n_samples,)).long()
        bool_tensor = torch.randint(low=0, high=2, size=(n_samples,)).to(
            TORCH_BOOL
        )
        int_tensor = torch.randint(low=0, high=10, size=(n_samples,)).to(
            torch.int32
        )
        one_hot = torch.nn.functional.one_hot(y).float()

        dataset = NamedTensorDataset(X, one_hot, bool_tensor, int_tensor)
        datasets.append((dataset, n_classes))

    return datasets


def test_accumulation_no_kv(generate_no_kv_datasets) -> None:
    """
    Check if all the data was accumulated correctly in LoaderMetricCallback.
    """
    for dataset, n_classes in generate_no_kv_datasets:
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=32)
        loaders = {"train": loader, "valid": loader}

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
    Check if all the kv-data accumulated in LoaderMetricCallback is valid:
    all input and output keys were accumulated in full, its' data types
    correspond to expected ones.
    """
    for dataset, n_classes in generate_kv_datasets:
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=32)
        loaders = {"train": loader, "valid": loader}

        model = DummyModel(output_dim=n_classes)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters())

        input_keys = ["features", "targets", "bool_field", "int_field"]
        output_keys = ["logits"]
        # we are not going to check metric's value, only accumulated data
        callback = dl.LoaderMetricCallback(
            input_key=input_keys,
            output_key=output_keys,
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

        # All the keys are accumulated
        for key_group, storage in (
            (input_keys, callback.input),
            (output_keys, callback.output),
        ):
            for key in key_group:
                assert key in storage

        expected_features = dataset.tensors[0]
        accumulated_features = torch.from_numpy(callback.input["features"])
        assert torch.allclose(expected_features, accumulated_features)
        assert accumulated_features.dtype == torch.float32

        expected_targets = dataset.tensors[1]
        accumulated_targets = torch.from_numpy(callback.input["targets"])
        assert (expected_targets == accumulated_targets).all()
        assert accumulated_targets.dtype == torch.float32

        expected_bool = dataset.tensors[2]
        accumulated_bool = torch.from_numpy(callback.input["bool_field"])
        assert (expected_bool == accumulated_bool).all()
        assert accumulated_bool.dtype == TORCH_BOOL

        expected_int = dataset.tensors[3]
        accumulated_int = torch.from_numpy(callback.input["int_field"])
        assert (expected_int == accumulated_int).all()
        assert accumulated_int.dtype == torch.int32
