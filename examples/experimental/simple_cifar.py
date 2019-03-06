from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from catalyst.dl.callbacks import PrecisionCallback, EarlyStoppingCallback, \
    InferCallback
from catalyst.dl.experiments import SupervisedRunner

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


def get_loaders():
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms
    )
    train_loader = DataLoader(
        train_set,
        batch_size=100,
        shuffle=True,
        num_workers=0
    )

    validation_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )

    return OrderedDict(train=train_loader, valid=validation_loader)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


loaders = get_loaders()
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[5,10],
    gamma=0.2)
runner = SupervisedRunner()

# training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
        PrecisionCallback(),
        EarlyStoppingCallback(patience=2, min_delta=0.01)
    ],
    verbose=True,
    logdir="./logs/01",
    n_epochs=500,
    main_metric="precision03",
    minimize_metric=False
)
print("-"*80)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logs/02",
    n_epochs=2
)
print("-"*80)

loaders = OrderedDict([("infer", loaders["train"])])
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[InferCallback(out_prefix="./logs/02/{suffix}.npy")],
)
print("-"*80)
