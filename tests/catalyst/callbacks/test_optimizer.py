# flake8: noqa
import torch
import torch.nn as nn

from catalyst.callbacks import backward
from catalyst.callbacks.backward import BackwardCallback
from catalyst.callbacks.optimizer import OptimizerCallback
from catalyst.engines.torch import CPUEngine


class DummyRunner:
    def __init__(
        self,
        loss_value: torch.tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.batch_metrics = {"loss": loss_value}
        self.is_train_loader = True
        self.model = model
        self.optimizer = optimizer
        # self.device = torch.device("cpu")
        self.engine = CPUEngine()


def test_zero_grad():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 3
    inp = torch.randn(batch_size, 10)
    target = torch.FloatTensor(batch_size, 2).uniform_()

    backward_callback = BackwardCallback(metric_key="loss")
    optimizer_callback = OptimizerCallback(metric_key="loss")

    loss1 = criterion(model(inp), target)
    loss1_value = loss1.detach().item()

    runner = DummyRunner(loss1, model, optimizer)

    for clb in [backward_callback, optimizer_callback]:
        clb.on_experiment_start(runner)
        clb.on_epoch_start(runner)
        clb.on_batch_end(runner)

    loss2 = criterion(model(inp), target)
    loss2_value = loss2.detach().item()

    runner.batch_metrics = {"loss": loss2}
    for clb in [backward_callback, optimizer_callback]:
        clb.on_epoch_start(runner)
        clb.on_batch_end(runner)

    assert loss1_value > loss2_value
