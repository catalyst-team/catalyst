from typing import Any, Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catalyst.contrib.dl.callbacks.loss_interpretation_callback import (
    LossInterpretationCallback,
)
from catalyst.core import IRunner


class IRunnerMock(IRunner):
    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        pass


def test_loss_interpretation_callback_skips_loader():
    runner = IRunnerMock()
    loss_interpretation_callback = LossInterpretationCallback(
        criterion=nn.CrossEntropyLoss(reduction="none"),
        loaders_to_skip=["valid"],
    )

    runner.loader_name = "valid"
    loss_interpretation_callback.on_loader_start(runner)


def test_loss_interpretation_callback_on_simple_example(tmp_path):
    runner = IRunnerMock()
    runner.loader_name = "valid"
    X = torch.rand(128, 3, 64, 64)
    n_classes = 10
    y = torch.randint(high=n_classes, size=(128,))
    dataset = TensorDataset(X, y)
    runner.logdir = Path(tmp_path / "./logdir_mock")
    runner.logdir.mkdir(exist_ok=True)
    bs = 64
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    runner.loaders = {"valid": loader}

    loss_interpretation_callback = LossInterpretationCallback(
        criterion=nn.CrossEntropyLoss(reduction="none")
    )

    loss_interpretation_callback.on_loader_start(runner)

    expected_best = torch.arange(10, 20)
    expected_worst = torch.arange(20, 30)
    logits = torch.zeros(len(y), n_classes)
    logits.scatter_(1, y.unsqueeze(1), 1)
    # Make logits absolutely correct
    logits[expected_best] *= 100
    # Make wrong logits in the predefined spot
    logits[expected_worst] *= -100
    batch_cnt = 0
    for X_batch, y_batch in iter(loader):
        start = batch_cnt
        end = batch_cnt + len(y_batch)
        runner.output = {"logits": logits[start:end]}
        runner.input = {"targets": y_batch}
        loss_interpretation_callback.on_batch_end(runner)
        batch_cnt += len(y_batch)

    loss_interpretation_callback.on_loader_end(runner)

    valid_interpretations = torch.load(
        runner.logdir / f"{runner.loader_name}_interpretations.pkl"
    )
    sorter = valid_interpretations["loss"].argsort()
    actual_best = valid_interpretations["indices"][sorter][:10]
    actual_worst = valid_interpretations["indices"][sorter][-10:][::-1]

    assert set(expected_best.numpy()) == (set(actual_best))
    assert set(expected_worst.numpy()) == (set(actual_worst))
