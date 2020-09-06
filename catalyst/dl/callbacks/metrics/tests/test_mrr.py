import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import (
    MRRCallback, SupervisedRunner, 
    SchedulerCallback
)

num_samples, num_features = 10_000, 10
n_classes = 10
X = torch.rand(num_samples, num_features)
y = torch.randint(0, n_classes, [num_samples])
loader = DataLoader(TensorDataset(X, y), batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# mdeol
model = torch.nn.Linear(num_features, n_classes)
criterion = torch.nn.CrossEntropyLoss()
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
    logdir="./logdir",
    num_epochs=2,
    verbose=True,
    callbacks=[MRRCallback, SchedulerCallback(reduced_metric="loss")]
)