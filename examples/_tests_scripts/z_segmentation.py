#!/usr/bin/env python
# coding: utf-8
# flake8: noqa
# isort:skip_file

# # Data

# In[ ]:

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# ! pip install tifffile

# In[ ]:

import tifffile as tiff

images = tiff.imread('./data/isbi/train-volume.tif')
masks = tiff.imread('./data/isbi/train-labels.tif')

data = list(zip(images, masks))

train_data = data[:2]
valid_data = data[:2]

# In[ ]:

import collections
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.contrib.nn.criterion import LovaszLossBinary, \
    LovaszLossMultiLabel, \
    LovaszLossMultiClass

bs = 1
num_workers = 0


def get_loaders(transform):
    open_fn = lambda x: {"features": x[0], "targets": x[1]}

    loaders = collections.OrderedDict()

    train_loader = utils.get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=transform,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=True
    )

    valid_loader = utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=transform,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=False
    )

    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


data_transform = transforms.Compose([
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / 255.).unsqueeze_(0)),
    Augmentor(
        dict_key="features",
        augment_fn=transforms.Normalize(
            (0.5,),
            (0.5,))),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / 255.).unsqueeze_(0))
])

loaders = get_loaders(data_transform)

# # Model

# In[ ]:

from catalyst.contrib.models.cv import Unet

# # Train

# In[ ]:

import torch
import torch.nn as nn
from catalyst.dl.runner import SupervisedDLRunner

# experiment setup
num_epochs = 2
logdir = "./logs/segmentation_notebook"

# model, criterion, optimizer
model = Unet(num_classes=1, in_channels=1, num_channels=32, num_blocks=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 20, 40], gamma=0.3
)

# model runner
runner = SupervisedDLRunner()

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

# # Inference

# In[ ]:

runner_out = runner.predict_loader(
    model, loaders["valid"], resume=f"{logdir}/checkpoints/best.pth"
)

# # Predictions visualization

# In[ ]:

import matplotlib.pyplot as plt
plt.style.use("ggplot")

# In[ ]:

sigmoid = lambda x: 1 / (1 + np.exp(-x))

for i, (input, output) in enumerate(zip(valid_data, runner_out)):
    image, mask = input

    threshold = 0.5

    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(image, 'gray')

    plt.subplot(1, 3, 2)
    output = sigmoid(output[0].copy())
    output = (output > threshold).astype(np.uint8)
    plt.imshow(output, 'gray')

    plt.subplot(1, 3, 3)
    plt.imshow(mask, 'gray')

    plt.show()

# lovasz LovaszLossBinary criterion

criterion = LovaszLossBinary()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# Multiclasses checks
model = Unet(num_classes=2, in_channels=1, num_channels=32, num_blocks=2)

# lovasz LovaszLossMultiClass criterion

data_transform = transforms.Compose([
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(
                x.copy().astype(np.float32) / 255.).unsqueeze_(0)),
    Augmentor(
        dict_key="features",
        augment_fn=transforms.Normalize(
            (0.5,),
            (0.5,))),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(
                x.copy().astype(np.float32) / 255.
            ))
])

loaders = get_loaders(data_transform)

criterion = LovaszLossMultiClass()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)

# lovasz LovaszLossMultiLabel criterion


def transform_targets(x):
    x1 = x.copy().astype(np.float32)[None]
    x2 = 255 - x.copy().astype(np.float32)[None]
    return np.vstack([x1, x2]) / 255.


data_transform = transforms.Compose([
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(
                x.copy().astype(np.float32) / 255.
            ).unsqueeze_(0)),
    Augmentor(
        dict_key="features",
        augment_fn=transforms.Normalize(
            (0.5,),
            (0.5,))),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(transform_targets(x)))
])

loaders = get_loaders(data_transform)

criterion = LovaszLossMultiLabel()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    check=True
)
