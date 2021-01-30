# flake8: noqa
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.callbacks.logging import CSVLogger
from catalyst.dl import SupervisedRunner


def test_logger():
    # data
    num_samples, num_features = int(1e4), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=[CSVLogger()],
        loaders=loaders,
        logdir="./logdir/test_csv",
        num_epochs=8,
        verbose=True,
    )
    assert os.path.exists("./logdir/test_csv/train_log/logs.csv")
    assert os.path.exists("./logdir/test_csv/valid_log/logs.csv")
    with open("./logdir/test_csv/train_log/logs.csv", "r") as log:
        length = 0
        for i, line in enumerate(log):
            if i == 0:
                assert "step,loss" in line
            length += 1
        assert length == 9
