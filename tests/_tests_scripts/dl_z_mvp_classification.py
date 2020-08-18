# flake8: noqa
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl

if os.getenv("USE_APEX", "0") != "0" or os.getenv("USE_DDP", "0") != "0":
    sys.exit()


# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples,) * num_classes).to(torch.int64)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

# model training
runner = dl.SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir="./logdir",
    num_epochs=3,
    check=True,
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)],
)
