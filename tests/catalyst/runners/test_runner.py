# flake8: noqa
from typing import Tuple
import shutil

import pytest

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from catalyst import dl
from catalyst.core.runner import IRunnerError
from catalyst.dl import Callback, CallbackOrder, SupervisedRunner


class DummyDataset(Dataset):
    """
    Dummy dataset.
    """

    features_dim: int = 4
    out_dim: int = 1

    def __len__(self):
        """
        Returns:
            dataset's length.
        """
        return 4

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            idx: index of sample

        Returns:
            dummy features and targets vector
        """
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


def run_train_with_empty_loader() -> None:
    """
    In this function we push loader to be empty because we
    use batch_size > len(dataset) and drop_last=True.
    """
    dataset = DummyDataset()
    model = nn.Linear(in_features=dataset.features_dim, out_features=dataset.out_dim)
    loader = DataLoader(dataset=dataset, batch_size=len(dataset) + 1, drop_last=True)
    runner = SupervisedRunner()
    runner.train(
        loaders={"train": loader},
        model=model,
        num_epochs=1,
        criterion=nn.BCEWithLogitsLoss(),
    )


def test_cathing_empty_loader() -> None:
    """
    We expect a error because loader is empty.
    """
    try:
        run_train_with_empty_loader()
    except IRunnerError:
        pass


def test_evaluation_loader_metrics() -> None:
    """
    Test if metrics in evaluate loader works properly.
    """
    dataset = DummyDataset()
    model = nn.Linear(in_features=dataset.features_dim, out_features=dataset.out_dim)
    loader = DataLoader(dataset=dataset, batch_size=1)
    callbacks = [
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,))
    ]
    runner = SupervisedRunner()
    runner.train(
        loaders={"train": loader, "valid": loader},
        model=model,
        num_epochs=1,
        criterion=nn.BCEWithLogitsLoss(),
        callbacks=callbacks,
    )
    runner_internal_metrics = runner.loader_metrics
    evaluate_loader_metrics = runner.evaluate_loader(loader=loader, callbacks=callbacks)
    assert runner_internal_metrics["accuracy01"] == evaluate_loader_metrics["accuracy01"]


def test_evaluation_loader_empty_model() -> None:
    """
    Test if there is no model was given, assertion raises.
    """
    with pytest.raises(AssertionError) as record:
        dataset = DummyDataset()
        loader = DataLoader(dataset=dataset, batch_size=1)
        callbacks = [
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,))
        ]
        runner = SupervisedRunner()
        runner.evaluate_loader(loader=loader, callbacks=callbacks, model=None)
        if not record:
            pytest.fail("Expected assertion bacuase model is empty!")


def test_evaluation_loader_custom_model() -> None:
    """
    Test if evaluate loader works with custom model.
    """
    dataset = DummyDataset()
    model = nn.Linear(in_features=dataset.features_dim, out_features=dataset.out_dim)
    loader = DataLoader(dataset=dataset, batch_size=1)
    callbacks = [
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,))
    ]
    runner = SupervisedRunner()

    runner.evaluate_loader(loader=loader, callbacks=callbacks, model=model)


def test_epoch_increasing():
    class IncreaseCheckerCallback(Callback):
        def __init__(self, attribute: str, start_value: int = None):
            super().__init__(CallbackOrder.Internal)
            self.attr = attribute
            self.prev = start_value

        def on_epoch_start(self, runner):
            if not hasattr(runner, self.attr):
                raise ValueError(f"There is no {self.attr} in runner!")
            value = getattr(runner, self.attr)
            if self.prev is not None:
                # print(
                #     f">>> '{self.attr}': "
                #     f"previous - {self.prev}, "
                #     f"current - {value}"
                # )
                assert self.prev < value
            self.prev = value

    # experiment_setup
    logdir = "./logs/core_runner"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    callbacks = [
        IncreaseCheckerCallback("epoch_step"),
        IncreaseCheckerCallback("batch_step"),
        IncreaseCheckerCallback("sample_step"),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=2,
        verbose=False,
        callbacks=callbacks,
    )

    shutil.rmtree(logdir, ignore_errors=True)
