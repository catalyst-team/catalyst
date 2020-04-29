# alias for https://catalyst-team.github.io/catalyst/info/distributed.html#case-3-best-practices-for-distributed-training # noqa: E501 W505
# flake8: noqa
# isort:skip_file
import os
import sys


if os.getenv("USE_APEX", "0") != "0" or os.getenv("USE_DDP", "0") != "1":
    sys.exit()


import torch
from torch.utils.data import TensorDataset

from catalyst.dl import SupervisedRunner, utils


def datasets_fn(num_features: int):
    """
    Docs.
    """
    X = torch.rand(int(1e4), num_features)
    y = torch.rand(X.shape[0])
    dataset = TensorDataset(X, y)
    return {"train": dataset, "valid": dataset}


def train():
    """
    Docs.
    """
    num_features = int(1e1)
    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    runner = SupervisedRunner()
    runner.train(
        model=model,
        datasets={
            "batch_size": 32,
            "num_workers": 1,
            "get_datasets_fn": datasets_fn,
            "num_features": num_features,
        },
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        logdir="./logs/example_3",
        num_epochs=8,
        verbose=True,
        distributed=False,
        check=True,
    )


utils.distributed_cmd_run(train)
