# flake8: noqa
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import SupervisedRunner

if os.getenv("USE_APEX", "0") != "0" or os.getenv("USE_DDP", "0") != "0":
    sys.exit()


# data
num_samples, num_features = int(32e3), int(1e1)
X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

runner = SupervisedRunner()
# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir="./logdir",
    num_epochs=8,
    verbose=True,
    check=True,
    load_best_on_end=True,
)
# model inference
for prediction in runner.predict_loader(loader=loader):
    assert prediction["logits"].cpu().detach().numpy().shape == (32, 1)
# model tracing
traced_model = runner.trace(loader=loader)
