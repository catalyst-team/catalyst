# alias for https://catalyst-team.github.io/catalyst/info/distributed.html#case-2-we-are-going-deeper # noqa: E501 W505

import torch
from torch.utils.data import TensorDataset

from catalyst.dl import SupervisedRunner

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(int(1e4), num_features)
y = torch.rand(X.shape[0])
dataset = TensorDataset(X, y)

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
        "train": dataset,
        "valid": dataset,
    },
    criterion=criterion,
    optimizer=optimizer,
    logdir="./logs/example_2",
    num_epochs=8,
    verbose=True,
    distributed=True,
    check=True,
)
