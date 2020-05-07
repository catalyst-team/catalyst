# alias for https://catalyst-team.github.io/catalyst/info/distributed.html#stage-1-i-just-want-distributed # noqa: E501 W505
# flake8: noqa
# isort:skip_file
import os
import sys


if os.getenv("USE_APEX", "0") != "0" or os.getenv("USE_DDP", "0") != "1":
    sys.exit()


import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import SupervisedRunner

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
    loaders=loaders,
    logdir="./logs/example_1",
    num_epochs=8,
    verbose=True,
    distributed=True,
)
