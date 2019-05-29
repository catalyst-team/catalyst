#!/usr/bin/env python
# coding: utf-8
# flake8: noqa

# # Data

# In[ ]:

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

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
from catalyst.data.augmentor import Augmentor
from catalyst.dl.utils import UtilsFactory

bs = 1
num_workers = 0

data_transform = transforms.Compose([
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / 255.).unsqueeze_(0)),
    Augmentor(
        dict_key="features",
        augment_fn=transforms.Normalize(
            (0.5, ),
            (0.5, ))),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / 255.).unsqueeze_(0))
])

open_fn = lambda x: {"features": x[0], "targets": x[1]}

loaders = collections.OrderedDict()

train_loader = UtilsFactory.create_loader(
    train_data,
    open_fn=open_fn,
    dict_transform=data_transform,
    batch_size=bs,
    num_workers=num_workers,
    shuffle=True
)

valid_loader = UtilsFactory.create_loader(
    valid_data,
    open_fn=open_fn,
    dict_transform=data_transform,
    batch_size=bs,
    num_workers=num_workers,
    shuffle=False
)

loaders["train"] = train_loader
loaders["valid"] = valid_loader

# # Model

# In[ ]:

from catalyst.contrib.models.segmentation import UNet

# # Train

# In[ ]:

import torch
import torch.nn as nn
from catalyst.dl.experiments import SupervisedRunner

# experiment setup
num_epochs = 2
logdir = "./logs/segmentation_notebook"

# model, criterion, optimizer
model = UNet(num_classes=1, in_channels=1, num_channels=32, num_blocks=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 20, 40], gamma=0.3
)

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

# # Inference

# In[ ]:

from catalyst.dl.callbacks import InferCallback, CheckpointCallback
loaders = collections.OrderedDict([("infer", loaders["valid"])])
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
)

# # Predictions visualization

# In[ ]:

import matplotlib.pyplot as plt
plt.style.use("ggplot")

# In[ ]:

sigmoid = lambda x: 1 / (1 + np.exp(-x))

for i, (input, output) in enumerate(
    zip(valid_data, runner.callbacks[1].predictions["logits"])
):
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

# In[ ]:
