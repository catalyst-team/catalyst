# flake8: noqa
import shutil

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import Callback, CallbackOrder, SupervisedRunner


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
    loaders = {
        "train": loader,
        "valid": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    callbacks = [
        IncreaseCheckerCallback("global_epoch_step"),
        IncreaseCheckerCallback("global_batch_step"),
        IncreaseCheckerCallback("global_sample_step"),
    ]

    # first stage
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

    # # second stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=3,
    #     verbose=False,
    #     callbacks=callbacks,
    # )
    #
    # # third stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=4,
    #     verbose=False,
    #     callbacks=callbacks,
    # )

    shutil.rmtree(logdir, ignore_errors=True)

    # # new exp
    # runner = SupervisedRunner()
    #
    # # first stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=2,
    #     verbose=False,
    #     callbacks=[
    #         IncreaseCheckerCallback("global_epoch_step"),
    #         IncreaseCheckerCallback("global_batch_step"),
    #         IncreaseCheckerCallback("global_sample_step"),
    #     ],
    # )
    #
    # # second stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=3,
    #     verbose=False,
    #     callbacks=[
    #         IncreaseCheckerCallback("global_epoch_step", 2),
    #         IncreaseCheckerCallback("global_batch_step", 626),
    #         IncreaseCheckerCallback("global_sample_step", 20_000),
    #     ],
    # )
    #
    # shutil.rmtree(logdir, ignore_errors=True)
