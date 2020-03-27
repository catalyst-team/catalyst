import collections
from numbers import Number

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from catalyst.contrib import registry
from catalyst.core import CriterionCallback, OptimizerCallback
from catalyst.dl import SupervisedRunner


@registry.Model
class _SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        return x


def _get_loaders(batch_size=1, num_workers=1):
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = CIFAR10(
        root="./dataset", train=True, download=True, transform=data_transform
    )
    trainset.data = trainset.data[:batch_size]
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    loaders = collections.OrderedDict(train=trainloader)

    return loaders


def test_save_model_grads():
    """
    Tests a feature of `OptimizerCallback` for saving model gradients
    """
    model = _SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    logdir = "./logs"
    loaders = _get_loaders(batch_size=4, num_workers=1)
    callbacks = collections.OrderedDict(
        loss=CriterionCallback(),
        optimizer=OptimizerCallback(save_model_grads=True)
    )
    runner = SupervisedRunner()
    runner.train(
        model, criterion, optimizer, loaders, logdir, callbacks=callbacks
    )

    prefix = callbacks["optimizer"].model_grad_norm_prefix

    for layer in ["conv1", "conv2", "fc1"]:
        for weights in ["weight", "bias"]:
            tag = f"{prefix}/{layer}/{weights}"
            assert tag in runner.state.batch_metrics
            assert isinstance(runner.state.batch_metrics[tag], Number)

    tag = f"{prefix}/total"
    assert tag in runner.state.batch_metrics
    assert isinstance(runner.state.batch_metrics[tag], Number)
