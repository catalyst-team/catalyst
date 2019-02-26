from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from catalyst.dl.callbacks import \
    LossCallback, OptimizerCallback, Callback, \
    SchedulerCallback, CheckpointCallback
from catalyst.dl.experiments.runner import SupervisedRunner
from catalyst.dl.state import RunnerState

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


class DictDatasetAdapter(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, target = self.dataset[index]

        return dict(image=image, targets=target)

    def __len__(self):
        return len(self.dataset)


def get_loaders():
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms
    )
    train_loader = DataLoader(
        DictDatasetAdapter(train_set),
        batch_size=100,
        shuffle=True,
        num_workers=0
    )

    validation_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms
    )

    validation_loader = DataLoader(
        DictDatasetAdapter(validation_set),
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


class VerboseCallback(Callback):
    def __init__(self):
        self.tqdm: tqdm = None
        self.step = 0

    def on_loader_start(self, state: RunnerState):
        self.step = 0
        self.tqdm = tqdm(total=state.loader_len)

    def on_batch_end(self, state: RunnerState):
        self.tqdm.update(self.step + 1)

    def on_loader_end(self, state: RunnerState):
        self.tqdm.close()
        self.tqdm = None
        self.step = 0


model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[2, 3],
    gamma=0.2)
runner = SupervisedRunner(model=model, input_key="image")

# training
runner.train(
    logdir="./logs/01",
    epochs=5,
    loaders=get_loaders(),
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=OrderedDict(
        tqdm=VerboseCallback(),
        loss=LossCallback(),
        optimizer=OptimizerCallback(),
        scheduler=SchedulerCallback(),
        saver=CheckpointCallback()
    )
)
