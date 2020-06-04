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


# example  10 - distibuted training with datasets preparation
# and fp16 support
# and utils.distributed_cmd_run
def train():
    X = torch.rand(int(1e4), 10)
    y = torch.rand(X.shape[0])
    model = Projector(X.shape[1])
    dataset = TensorDataset(X, y)
    # loader = DataLoader(dataset, batch_size=32, num_workers=1)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        # loaders={"train": loader, "valid": loader},
        datasets={
            "batch_size": 32,
            "num_workers": 1,
            "train": dataset,
            "valid": dataset,
        },
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters()),
        logdir="logs/log_example_10",
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
