# flake8: noqa
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import (
    Callback,
    CallbackOrder,
    CriterionCallback,
    PeriodicLoaderCallback,
    SupervisedRunner,
)


class BestStateCheckerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.External)
        self.valid_loader = None

    def on_stage_start(self, runner: "IRunner") -> None:
        self.valid_loader = copy.copy(runner.valid_loader)

    def on_epoch_end(self, runner: "IRunner") -> None:
        if self.valid_loader not in runner.loaders and runner.epoch > 1:
            assert (
                not runner.is_best_valid
            ), f"Epochs (epoch={runner.epoch}) without valid loader can't be best!"
        else:
            assert runner.valid_metrics[runner.main_metric] is not None


# experiment_setup
logdir = "./logs/core_runner"

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    "train": loader,
    "valid": loader,
}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
runner = SupervisedRunner()


# first stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=5,
    verbose=False,
    callbacks=[PeriodicLoaderCallback(valid=2), BestStateCheckerCallback()],
)

# second stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=6,
    verbose=False,
    callbacks=[PeriodicLoaderCallback(valid=3), BestStateCheckerCallback()],
)

# third stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=6,
    verbose=False,
    callbacks=[PeriodicLoaderCallback(valid=4), BestStateCheckerCallback()],
)
