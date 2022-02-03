# flake8: noqa
from typing import Tuple

import torch
from torch import nn

from catalyst import dl


class DummyRunner(dl.Runner):
    def handle_batch(self, batch: Tuple[torch.Tensor]):
        pass


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def test_soft_update():

    model = nn.ModuleDict(
        {
            "target": nn.Linear(10, 10, bias=False),
            "source": nn.Linear(10, 10, bias=False),
        }
    )
    set_requires_grad(model, False)
    model["target"].weight.data.fill_(0)

    runner = DummyRunner(model=model)
    runner.is_train_loader = True

    soft_update = dl.SoftUpdateCallaback(
        target_model="target",
        source_model="source",
        tau=0.1,
        scope="on_batch_end",
    )
    soft_update.on_batch_end(runner)

    checks = (
        (
            (0.1 * runner.model["source"].weight.data)
            == runner.model["target"].weight.data
        )
        .flatten()
        .tolist()
    )

    assert all(checks)


def test_soft_update_not_work():

    model = nn.ModuleDict(
        {
            "target": nn.Linear(10, 10, bias=False),
            "source": nn.Linear(10, 10, bias=False),
        }
    )
    set_requires_grad(model, False)
    model["target"].weight.data.fill_(0)

    runner = DummyRunner(model=model)
    runner.is_train_loader = True

    soft_update = dl.SoftUpdateCallaback(
        target_model="target",
        source_model="source",
        tau=0.1,
        scope="on_batch_start",
    )
    soft_update.on_batch_end(runner)

    checks = (runner.model["target"].weight.data == 0).flatten().tolist()

    assert all(checks)
