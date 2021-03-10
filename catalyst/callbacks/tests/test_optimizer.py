# flake8: noqa
import random

import torch
import torch.nn as nn

from catalyst.callbacks import OptimizerCallback
from catalyst.engines.torch import DeviceEngine


class DummyRunner:
    def __init__(
        self, loss_value: torch.tensor, model: nn.Module, optimizer: torch.optim.Optimizer
    ):
        self.batch_metrics = {"loss": loss_value}
        self.is_train_loader = True
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cpu")
        self.engine = DeviceEngine("cpu")


def test_zero_grad():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 3
    inp = torch.randn(batch_size, 10)
    target = torch.FloatTensor(batch_size, 2).uniform_()

    callback = OptimizerCallback(metric_key="loss")

    loss1 = criterion(model(inp), target)
    loss1_value = loss1.detach().item()

    runner = DummyRunner(loss1, model, optimizer)

    callback.on_stage_start(runner)
    callback.on_epoch_start(runner)
    callback.on_batch_end(runner)

    loss2 = criterion(model(inp), target)
    loss2_value = loss2.detach().item()

    runner.batch_metrics = {"loss": loss2}
    callback.on_epoch_start(runner)
    callback.on_batch_end(runner)

    assert loss1_value > loss2_value


# def test_fast_zero_grad():
#     model = nn.Linear(10, 2)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.BCEWithLogitsLoss()
#
#     batch_size = 3
#     inp = torch.randn(batch_size, 10)
#     target = torch.FloatTensor(batch_size, 2).uniform_()
#
#     callback = OptimizerCallback(metric_key="loss", use_fast_zero_grad=True)
#
#     loss1 = criterion(model(inp), target)
#     loss1_value = loss1.detach().item()
#
#     runner = DummyRunner(loss1, optimizer)
#
#     callback.on_stage_start(runner)
#     callback.on_epoch_start(runner)
#     callback.on_batch_end(runner)
#
#     loss2 = criterion(model(inp), target)
#     loss2_value = loss2.detach().item()
#
#     runner.batch_metrics = {"loss": loss2}
#     callback.on_epoch_start(runner)
#     callback.on_batch_end(runner)
#
#     assert loss1_value > loss2_value
