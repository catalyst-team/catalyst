#!/usr/bin/env python
# coding: utf-8
# flake8: noqa

# # Data

# In[ ]:

import collections
import torch
import torchvision
import torchvision.transforms as transforms

bs = 32
num_workers = 0

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))]
)

loaders = collections.OrderedDict()

trainset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=data_transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs, shuffle=True, num_workers=num_workers
)

testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=data_transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs, shuffle=False, num_workers=num_workers
)

loaders["train"] = trainloader
loaders["valid"] = testloader

# # Model

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # Intro
# @TODO

# In[ ]:

# for graphs use `tensorboard --logdir=./logs`

# In[ ]:

from catalyst.dl import utils

# # Setup 1 - typical training

# In[ ]:

from catalyst.dl.runner import SupervisedRunner

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_1"

# model, criterion, optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# In[ ]:

# you can use plotly and tensorboard to plot metrics inside jupyter
# by default it only plots loss
# logs_plot = utils.plot_metrics(logdir=logdir)

# # Setup 2 - training with scheduler

# In[ ]:

from catalyst.dl.runner import SupervisedRunner

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_2"

# model, criterion, optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# any Pytorch scheduler supported
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2
)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# # Setup 3 - training with early stop

# In[ ]:

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_3"

# model, criterion, optimizer, scheduler
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3, 8], gamma=0.3
)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[EarlyStoppingCallback(patience=2, min_delta=0.01)],
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# In[ ]:

# logs_plot = utils.plot_metrics(logdir=logdir, metrics=["loss", "_base/lr"])

# # Setup 4 - training with additional metrics

# In[ ]:

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_4"

# model, criterion, optimizer, scheduler
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3, 8], gamma=0.3
)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
        AccuracyCallback(accuracy_args=[1, 3, 5]),
        EarlyStoppingCallback(patience=2, min_delta=0.01)
    ],
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# In[ ]:

# logs_plot = utils.plot_metrics(
#     logdir=logdir, metrics=["loss", "accuracy01", "accuracy03", "_base/lr"]
# )

# # Setup 5 - training with 1cycle

# In[ ]:

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback
from catalyst.contrib.schedulers import OneCycleLR

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_5"

# model, criterion, optimizer, scheduler
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = OneCycleLR(
    optimizer,
    num_steps=num_epochs,
    lr_range=(0.005, 0.00005),
    warmup_steps=2,
    momentum_range=(0.85, 0.95)
)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
        AccuracyCallback(accuracy_args=[1, 3, 5]),
        EarlyStoppingCallback(patience=2, min_delta=0.01),
    ],
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# In[ ]:

# logs_plot = utils.plot_metrics(
#     logdir=logdir,
#     step="batch",
#     metrics=["loss", "accuracy01", "_base/lr", "_base/momentum"]
# )

# # Setup 6 - pipeline check

# In[ ]:

from catalyst.dl.runner import SupervisedRunner

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_6"

# model, criterion, optimizer, scheduler
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3, 8], gamma=0.3
)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True  # here is the trick
)

# # Setup 7 - multi-stage training

# In[ ]:

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback

# experiment setup
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_7"

# model, criterion, optimizer, scheduler
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3, 8], gamma=0.3
)

# model runner
runner = SupervisedRunner()

# model training - 1
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
        AccuracyCallback(accuracy_args=[1, 3, 5]),
        EarlyStoppingCallback(patience=2, min_delta=0.01)
    ],
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# model training - 2
num_epochs = 2
logdir = "./logs/cifar_simple_notebook_8"
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# # Setup 8 - loader inference

# In[ ]:

from catalyst.dl.callbacks import InferCallback
loaders = collections.OrderedDict([("infer", loaders["train"])])
runner.infer(
    model=model, loaders=loaders, callbacks=[InferCallback()], check=True
)

# In[ ]:

runner.callbacks[0].predictions["logits"].shape

# # Setup 9 - batch inference

# In[ ]:

features, targets = next(iter(loaders["infer"]))

# In[ ]:

features.shape

# In[ ]:

runner_in = {runner.input_key: features}
runner_out = runner.predict_batch(runner_in)

# In[ ]:

runner_out["logits"].shape

# In[ ]:
