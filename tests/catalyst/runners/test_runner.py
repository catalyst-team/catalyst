from typing import Tuple

import pytest

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from catalyst import dl
from catalyst.core.runner import RunnerError
from catalyst.dl import SupervisedRunner


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
        loaders={"train": loader}, model=model, num_epochs=1, criterion=nn.BCEWithLogitsLoss()
    )


def test_cathing_empty_loader() -> None:
    """
    We expect a error because loader is empty.
    """
    try:
        run_train_with_empty_loader()
    except RunnerError:
        pass


def test_evaluation_loader_metrics() -> None:
    """
    Test if metrics in evaluate loader works properly.
    """
    dataset = DummyDataset()
    model = nn.Linear(in_features=dataset.features_dim, out_features=dataset.out_dim)
    loader = DataLoader(dataset=dataset, batch_size=1)
    callbacks = [dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1,))]
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
    assert runner_internal_metrics["accuracy"] == evaluate_loader_metrics["accuracy"]


def test_evaluation_loader_empty_model() -> None:
    """
    Test if there is no model was given, assertion raises.
    """
    with pytest.raises(AssertionError) as record:
        dataset = DummyDataset()
        loader = DataLoader(dataset=dataset, batch_size=1)
        callbacks = [dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1,))]
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
    callbacks = [dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1,))]
    runner = SupervisedRunner()

    runner.evaluate_loader(loader=loader, callbacks=callbacks, model=model)
