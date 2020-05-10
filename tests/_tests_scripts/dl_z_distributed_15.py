# flake8: noqa
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from catalyst import dl, utils


class Projector(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


def datasets_fn(num_features: int):
    """
    Datasets closure.

    Args:
        num_features: number of features for dataset creation.
    """
    X = torch.rand(int(1e4), num_features)
    y = torch.rand(X.shape[0])
    dataset = TensorDataset(X, y)
    return {"train": dataset, "valid": dataset}


# example 14 - distibuted training with datasets closure
# and fp16 support
# and utils.distributed_cmd_run
def train():
    num_features = 10
    model = Projector(num_features)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        # loaders={"train": loader, "valid": loader},
        datasets={
            "batch_size": 32,
            "num_workers": 1,
            "get_datasets_fn": datasets_fn,
            "num_features": num_features,
        },
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters()),
        logdir="logs/log_example_15",
        num_epochs=10,
        verbose=True,
        check=True,
        fp16=True,
        distributed=False,
    )


def main():
    utils.distributed_cmd_run(train)


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "1" and os.getenv("USE_DDP", "0") == "1":
        main()
