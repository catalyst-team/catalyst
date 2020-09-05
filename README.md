<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated deep learning R&D**

[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

![codestyle](https://github.com/catalyst-team/catalyst/workflows/codestyle/badge.svg?branch=master&event=push)
![catalyst](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
![catalyst-cv](https://github.com/catalyst-team/catalyst/workflows/catalyst-cv/badge.svg?branch=master&event=push)
![catalyst-nlp](https://github.com/catalyst-team/catalyst/workflows/catalyst-nlp/badge.svg?branch=master&event=push)
[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)

[![python](https://img.shields.io/badge/python_3.6-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
[![python](https://img.shields.io/badge/python_3.7-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
[![python](https://img.shields.io/badge/python_3.8-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)

[![os](https://img.shields.io/badge/Linux-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
[![os](https://img.shields.io/badge/OSX-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
[![os](https://img.shields.io/badge/WSL-passing-success)](https://github.com/catalyst-team/catalyst/workflows/catalyst/badge.svg?branch=master&event=push)
</div>

PyTorch framework for Deep Learning research and development.
It focuses on reproducibility, rapid experimentation, and codebase reuse 
so you can create something new rather than write another regular train loop.
<br/> Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - accelerated deep learning R&D
- [Reaction](https://github.com/catalyst-team/reaction) - convenient deep learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

----

## Getting started

```bash
pip install -U catalyst
```

```python
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

model = torch.nn.Linear(28 * 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))

        loss = F.cross_entropy(y_hat, y)
        accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
# model training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logs",
    num_epochs=5,
    verbose=True,
    load_best_on_end=True,
)
# model inference
for prediction in runner.predict_loader(loader=loaders["valid"]):
    assert prediction.detach().cpu().numpy().shape[-1] == 10
# model tracing
traced_model = runner.trace(loader=loaders["valid"])
```

### Step by step guide
1. Start with [Catalyst 101 — Accelerated PyTorch](https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92?source=friends_link&sk=d3dd9b2b23500eca046361187b4619ff) introduction.
2. Check [minimal examples](#minimal-examples).
3. Try [notebook tutorials with Google Colab](#tutorials).
4. Read [blogposts](#blogposts) with use-cases and guides (and Config API intro).
5. Go through advanced  [classification](https://github.com/catalyst-team/classification), [detection](https://github.com/catalyst-team/detection) and [segmentation](https://github.com/catalyst-team/segmentation) pipelines with Config API. More pipelines available under [projects section](#projects). 
6. Want more? See [Alchemy](https://github.com/catalyst-team/alchemy) and [Reaction](https://github.com/catalyst-team/reaction) packages.
7. For Catalyst.RL introduction, please follow [Catalyst.RL repo](https://github.com/catalyst-team/catalyst-rl).


## Table of Contents
- [Overview](#overview)
  * [Installation](#installation)
  * [Minimal examples](#minimal-examples)
  * [Features](#features)
  * [Structure](#structure)
  * [Tests](#tests)
- [Catalyst](#catalyst)
  * [Tutorials](#tutorials)
  * [Blogposts](#blogposts)
  * [Docs](#docs)
  * [Projects](#projects)
  * [Talks](#talks)
- [Community](#community)
  * [Contribution guide](#contribution-guide)
  * [User feedback](#user-feedback)
  * [Acknowledgments](#acknowledgments)
  * [Trusted by](#trusted-by)
  * [Supported by](#supported-by)
  * [Citation](#citation)


## Overview
Catalyst helps you write compact
but full-featured Deep Learning pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.


### Installation

Common installation:
```bash
pip install -U catalyst
```

<details>
<summary>Specific versions with additional requirements</summary>
<p>

```bash
pip install catalyst[cv]         # installs CV-based catalyst
pip install catalyst[nlp]        # installs NLP-based catalyst
pip install catalyst[ecosystem]  # installs Catalyst.Ecosystem
# and master version installation
pip install git+https://github.com/catalyst-team/catalyst@master --upgrade
```
</p>
</details>

Catalyst is compatible with: Python 3.6+. PyTorch 1.1+. <br/>
Tested on Ubuntu 16.04/18.04/20.04, macOS 10.15, Windows 10 and Windows Subsystem for Linux.


### Minimal Examples

<details>
<summary>ML - linear regression</summary>
<p>

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner

# data
num_samples, num_features = int(1e4), int(1e1)
X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 1)
criterion = torch.nn.MSELoss()
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
    num_epochs=8,
    verbose=True,
)
```
</p>
</details>


<details>
<summary>ML - multi-class classification</summary>
<p>

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

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
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)]
)
```
</p>
</details>


<details>
<summary>ML - multi-label classification</summary>
<p>

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes)
criterion = torch.nn.BCEWithLogitsLoss()
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
    callbacks=[dl.MultiLabelAccuracyCallback(threshold=0.5)]
)
```
</p>
</details>


<details>
<summary>CV - MNIST classification</summary>
<p>

```python
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

model = torch.nn.Linear(28 * 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))

        loss = F.cross_entropy(y_hat, y)
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }
        
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer, 
    loaders=loaders, 
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - classification with AutoEncoder</summary>
<p>

```python
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

class ClassifyAE(nn.Module):

    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hid_features), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(hid_features, in_features), nn.Sigmoid())
        self.clf = nn.Linear(hid_features, out_features)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.clf(z)
        x_ = self.decoder(z)
        return y_hat, x_

model = ClassifyAE(28 * 28, 128, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_ = self.model(x)

        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss = loss_clf + loss_ae
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - classification with Variational AutoEncoder</summary>
<p>

```python
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

LOG_SCALE_MAX = 2
LOG_SCALE_MIN = -10

def normal_sample(mu, sigma):
    return mu + sigma * torch.randn_like(sigma)

def normal_logprob(mu, sigma, z):
    normalization_constant = (-sigma.log() - 0.5 * np.log(2 * np.pi))
    square_term = -0.5 * ((z - mu) / sigma)**2
    logprob_vec = normalization_constant + square_term
    logprob = logprob_vec.sum(1)
    return logprob

class ClassifyVAE(torch.nn.Module):

    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = torch.nn.Linear(in_features, hid_features * 2)
        self.decoder = nn.Sequential(nn.Linear(hid_features, in_features), nn.Sigmoid())
        self.clf = torch.nn.Linear(hid_features, out_features)

    def forward(self, x, deterministic=False):
        z = self.encoder(x)
        bs, z_dim = z.shape

        loc, log_scale = z[:, :z_dim // 2], z[:, z_dim // 2:]
        log_scale = torch.clamp(log_scale, LOG_SCALE_MIN, LOG_SCALE_MAX)
        scale = torch.exp(log_scale)
        z_ = loc if deterministic else normal_sample(loc, scale)
        z_logprob = normal_logprob(loc, scale, z_)
        z_ = z_.view(bs, -1)
        x_ = self.decoder(z_)
        y_hat = self.clf(z_)

        return y_hat, x_, z_logprob, loc, log_scale

model = ClassifyVAE(28 * 28, 64, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_, z_logprob, loc, log_scale = self.model(x)

        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss_kld = -0.5 * torch.mean(1 + log_scale - loc.pow(2) - log_scale.exp()) * 0.1
        loss_logprob = torch.mean(z_logprob) * 0.01
        loss = loss_clf + loss_ae + loss_kld + loss_logprob
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss_kld": loss_kld,
            "loss_logprob": loss_logprob,
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - segmentation with classification auxiliary task</summary>
<p>

```python
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

class ClassifyUnet(nn.Module):

    def __init__(self, in_channels, in_hw, out_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.Tanh())
        self.decoder = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.clf = nn.Linear(in_channels * in_hw * in_hw, out_features)

    def forward(self, x):
        z = self.encoder(x)
        z_ = z.view(z.size(0), -1)
        y_hat = self.clf(z_)
        x_ = self.decoder(z)
        return y_hat, x_

model = ClassifyUnet(1, 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
        y_hat, x_ = self.model(x_noise)

        loss_clf = F.cross_entropy(y_hat, y)
        iou = metrics.iou(x_, x)
        loss_iou = 1 - iou
        loss = loss_clf + loss_iou
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_iou": loss_iou,
            "loss": loss,
            "iou": iou,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }
        
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer, 
    loaders=loaders, 
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - MNIST with Metric Learning</summary>
<p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xcob6Y2W0O1JiN-juoF1YfJMJsScCVhV?usp=sharing)

```python
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import data, dl, utils
from catalyst.contrib import datasets, models, nn
import catalyst.data.cv.transforms.torch as t


# 1. train and valid datasets
dataset_root = "."
transforms = t.Compose([t.ToTensor(), t.Normalize((0.1307,), (0.3081,))])

dataset_train = datasets.MnistMLDataset(root=dataset_root, train=True, download=True, transform=transforms)
sampler = data.BalanceBatchSampler(labels=dataset_train.get_labels(), p=10, k=10)
train_loader = DataLoader(dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size)

dataset_val = datasets.MnistQGDataset(root=dataset_root, transform=transforms, gallery_fraq=0.2)
val_loader = DataLoader(dataset=dataset_val, batch_size=1024)

# 2. model and optimizer
model = models.SimpleConv(features_dim=16)
optimizer = Adam(model.parameters(), lr=0.001)

# 3. criterion with triplets sampling
sampler_inbatch = data.HardTripletsSampler(norm_required=False)
criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

# 4. training with catalyst Runner
callbacks = [
    dl.ControlFlowCallback(dl.CriterionCallback(), loaders="train"),
    dl.ControlFlowCallback(dl.CMCScoreCallback(topk_args=[1]), loaders="valid"),
    dl.PeriodicLoaderCallback(valid=100),
]

runner = dl.SupervisedRunner(device=utils.get_device())
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders={"train": train_loader, "valid": val_loader},
    minimize_metric=False,
    verbose=True,
    valid_loader="valid",
    num_epochs=200,
    main_metric="cmc01",
)   
```
</p>
</details>

<details>
<summary>GAN - MNIST, flatten version</summary>
<p>

```python
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten, GlobalMaxPool2d, Lambda

latent_dim = 128
generator = nn.Sequential(
    # We want to generate 128 coefficients to reshape into a 7x7x128 map
    nn.Linear(128, 128 * 7 * 7),
    nn.LeakyReLU(0.2, inplace=True),
    Lambda(lambda x: x.view(x.size(0), 128, 7, 7)),
    nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
    nn.LeakyReLU(0.2, inplace=True),
    nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 1, (7, 7), padding=3),
    nn.Sigmoid(),
)
discriminator = nn.Sequential(
    nn.Conv2d(1, 64, (3, 3), stride=(2, 2), padding=1),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1),
    nn.LeakyReLU(0.2, inplace=True),
    GlobalMaxPool2d(),
    Flatten(),
    nn.Linear(128, 1)
)

model = {"generator": generator, "discriminator": discriminator}
optimizer = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
}
loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        real_images, _ = batch
        batch_metrics = {}
        
        # Sample random points in the latent space
        batch_size = real_images.shape[0]
        random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
        
        # Decode them to fake images
        generated_images = self.model["generator"](random_latent_vectors).detach()
        # Combine them with real images
        combined_images = torch.cat([generated_images, real_images])
        
        # Assemble labels discriminating real from fake images
        labels = torch.cat([
            torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))
        ]).to(self.device)
        # Add random noise to the labels - important trick!
        labels += 0.05 * torch.rand(labels.shape).to(self.device)
        
        # Train the discriminator
        predictions = self.model["discriminator"](combined_images)
        batch_metrics["loss_discriminator"] = \
          F.binary_cross_entropy_with_logits(predictions, labels)
        
        # Sample random points in the latent space
        random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((batch_size, 1)).to(self.device)
        
        # Train the generator
        generated_images = self.model["generator"](random_latent_vectors)
        predictions = self.model["discriminator"](generated_images)
        batch_metrics["loss_generator"] = \
          F.binary_cross_entropy_with_logits(predictions, misleading_labels)
        
        self.batch_metrics.update(**batch_metrics)

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer,
    loaders=loaders,
    callbacks=[
        dl.OptimizerCallback(
            optimizer_key="generator", 
            metric_key="loss_generator"
        ),
        dl.OptimizerCallback(
            optimizer_key="discriminator", 
            metric_key="loss_discriminator"
        ),
    ],
    main_metric="loss_generator",
    num_epochs=20,
    verbose=True,
    logdir="./logs_gan",
)
```
</p>
</details>

<details>
<summary>ML - multi-class classification (fp16 training version)</summary>
<p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q8BPg1XpQn2J5vWV9OYKSBo-k9wA2jYS?usp=sharing)

```python
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

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
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)],
    fp16=True,
)
```
</p>
</details>

<details>
<summary>ML - multi-class classification (advanced fp16 training version)</summary>
<p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q8BPg1XpQn2J5vWV9OYKSBo-k9wA2jYS?usp=sharing)

```python
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

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
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)],
    fp16=dict(opt_level="O1"),
)
```
</p>
</details>

<details>
<summary>ML - Linear Regression (distributed training version)</summary>
<p>

```python
#!/usr/bin/env python
import torch
from torch.utils.data import TensorDataset
from catalyst.dl import SupervisedRunner, utils

def datasets_fn(num_features: int):
    X = torch.rand(int(1e4), num_features)
    y = torch.rand(X.shape[0])
    dataset = TensorDataset(X, y)
    return {"train": dataset, "valid": dataset}

def train():
    num_features = int(1e1)
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
            "get_datasets_fn": datasets_fn,
            "num_features": num_features,  # will be passed to datasets_fn
        },
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        logdir="./logs/example_distributed_ml",
        num_epochs=8,
        verbose=True,
        distributed=False,
    )

utils.distributed_cmd_run(train)
```
</p>
</details>

<details>
<summary>CV - classification with AutoEncoder (distributed training version)</summary>
<p>

```python
#!/usr/bin/env python
import os
import torch
from torch import nn
from torch.nn import functional as F
from catalyst import dl, utils
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

class ClassifyAE(nn.Module):

    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hid_features), nn.Tanh())
        self.decoder = nn.Linear(hid_features, in_features)
        self.clf = nn.Linear(hid_features, out_features)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.clf(z)
        x_ = self.decoder(z)
        return y_hat, x_

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_ = self.model(x)

        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss = loss_clf + loss_ae
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

def datasets_fn():
    dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
    return {"train": dataset, "valid": dataset}

def train():
    model = ClassifyAE(28 * 28, 128, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        datasets={
            "batch_size": 32,
            "num_workers": 1,
            "get_datasets_fn": datasets_fn,
        },
        logdir="./logs/distributed_ae",
        num_epochs=8,
        verbose=True,
    )

utils.distributed_cmd_run(train)
```
</p>
</details>

<details>
<summary>ML - multi-class classification (TPU version)</summary>
<p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AhvNzTRb3gd3AYhzUfm3dzw8TddlsfhD?usp=sharing)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# device (TPU > GPU > CPU)
device = utils.get_device()  # <--------- TPU device

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

# model training
runner = dl.SupervisedRunner(device=device)
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logdir",
    num_epochs=3,
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)]
)
```
</p>
</details>


### Features
- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – all source code and environment variables will be saved.
- Callbacks – reusable train/inference pipeline parts with easy customization.
- Training stages support.
- Deep Learning best practices - SWA, AdamW, Ranger optimizer, OneCycle, and more.
- Developments best practices - fp16 support, distributed training, slurm support.


### Structure
- **core** - framework core with main abstractions - 
    Experiment, Runner and Callback.
- **data** - useful tools and scripts for data processing.
- **dl** – runner for training and inference,
   all of the classic ML and CV/NLP/RecSys metrics
   and a variety of callbacks for training, validation
   and inference of neural networks.
- **tools** - extra tools for Deep Learning research, class-based helpers.   
- **utils** - typical utils for Deep Learning research, function-based helpers.
- **contrib** - additional modules contributed by Catalyst users.


### Tests
All Catalyst code, features and pipelines [are fully tested](./tests) 
with our own [catalyst-codestyle](https://github.com/catalyst-team/codestyle).

In fact, we train a number of different models for various of tasks - 
image classification, image segmentation, text classification, GANs training 
and much more.
During the tests, we compare their convergence metrics in order to verify 
the correctness of the training procedure and its reproducibility.

As a result, Catalyst provides fully tested and reproducible 
best practices for your deep learning research.


## Catalyst

### Tutorials
- [Customizing what happens in `train`](./examples/notebooks/customizing_what_happens_in_train.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/customizing_what_happens_in_train.ipynb)
- [Demo with minimal examples](./examples/notebooks/demo.ipynb) for ML, CV, NLP, GANs and RecSys [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb)
- Detailed [classification tutorial](./examples/notebooks/classification-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb)
- Advanced [segmentation tutorial](./examples/notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)
- Metric Learning tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xcob6Y2W0O1JiN-juoF1YfJMJsScCVhV?usp=sharing)
- Catalyst with Google TPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AhvNzTRb3gd3AYhzUfm3dzw8TddlsfhD?usp=sharing)


### Blogposts

- [Catalyst 101 — Accelerated PyTorch](https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92?source=friends_link&sk=d3dd9b2b23500eca046361187b4619ff)
- [BERT Distillation with Catalyst](https://medium.com/pytorch/bert-distillation-with-catalyst-c6f30c985854?source=friends_link&sk=1a28469ac8c0e6e6ad35bd26dfd95dd9)
- [Metric Learning with Catalyst](https://medium.com/pytorch/metric-learning-with-catalyst-8c8337dfab1a?source=friends_link&sk=320b95f9b2a9074aab8d916ed78912d6)
- [Distributed training best practices](https://catalyst-team.github.io/catalyst/info/distributed.html)
- [Addressing the Cocktail Party Problem using PyTorch](https://medium.com/pytorch/addressing-the-cocktail-party-problem-using-pytorch-305fb74560ea)
- [Beyond fashion: Deep Learning with Catalyst (Config API)](https://evilmartians.com/chronicles/beyond-fashion-deep-learning-with-catalyst)
- [Tutorial from Notebook API to Config API (RU)](https://github.com/Bekovmi/Segmentation_tutorial)


### Docs

- [master](https://catalyst-team.github.io/catalyst/)
- [20.08.2](https://catalyst-team.github.io/catalyst/v20.08.2/index.html)
- [20.07](https://catalyst-team.github.io/catalyst/v20.07/index.html) - [dev blog: 20.07 release](https://medium.com/pytorch/catalyst-dev-blog-20-07-release-fb489cd23e14?source=friends_link&sk=7ab92169658fe9a9e1c44068f28cc36c)
- [20.06](https://catalyst-team.github.io/catalyst/v20.06/index.html)
- [20.05](https://catalyst-team.github.io/catalyst/v20.05/index.html), [20.05.1](https://catalyst-team.github.io/catalyst/v20.05.1/index.html)
- [20.04](https://catalyst-team.github.io/catalyst/v20.04/index.html), [20.04.1](https://catalyst-team.github.io/catalyst/v20.04.1/index.html), [20.04.2](https://catalyst-team.github.io/catalyst/v20.04.2/index.html)


### Projects

#### Examples, notebooks and starter kits
- [CamVid Segmentation Example](https://github.com/BloodAxe/Catalyst-CamVid-Segmentation-Example) - Example of semantic segmentation for CamVid dataset
- [Notebook API tutorial for segmentation in Understanding Clouds from Satellite Images Competition](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools/)
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://github.com/Scitator/learning-to-move-starter-kit) – starter kit
- [Catalyst.RL - NeurIPS 2019: Animal-AI Olympics](https://github.com/Scitator/animal-olympics-starter-kit) - starter kit
- [Inria Segmentation Example](https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example) - An example of training segmentation model for Inria Sattelite Segmentation Challenge
- [iglovikov_segmentation](https://github.com/ternaus/iglovikov_segmentation) - Semantic segmentation pipeline using Catalyst

#### Competitions
- [Kaggle Quick, Draw! Doodle Recognition Challenge](https://github.com/ngxbac/Kaggle-QuickDraw) - 11th place solution
- [Catalyst.RL - NeurIPS 2018: AI for Prosthetics Challenge](https://github.com/Scitator/neurips-18-prosthetics-challenge) – 3rd place solution
- [Kaggle Google Landmark 2019](https://github.com/ngxbac/Kaggle-Google-Landmark-2019) - 30th place solution
- [iMet Collection 2019 - FGVC6](https://github.com/ngxbac/Kaggle-iMet) - 24th place solution
- [ID R&D Anti-spoofing Challenge](https://github.com/bagxi/idrnd-anti-spoofing-challenge-solution) - 14th place solution
- [NeurIPS 2019: Recursion Cellular Image Classification](https://github.com/ngxbac/Kaggle-Recursion-Cellular) - 4th place solution
- [MICCAI 2019: Automatic Structure Segmentation for Radiotherapy Planning Challenge 2019](https://github.com/ngxbac/StructSeg2019) 
  * 3rd place solution for `Task 3: Organ-at-risk segmentation from chest CT scans`
  * and 4th place solution for `Task 4: Gross Target Volume segmentation of lung cancer`
- [Kaggle Seversteal steel detection](https://github.com/bamps53/kaggle-severstal) - 5th place solution
- [RSNA Intracranial Hemorrhage Detection](https://github.com/ngxbac/Kaggle-RSNA) - 5th place solution
- [APTOS 2019 Blindness Detection](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection) – 7th place solution
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://github.com/Scitator/run-skeleton-run-in-3d) – 2nd place solution
- [xView2 Damage Assessment Challenge](https://github.com/BloodAxe/xView2-Solution) - 3rd place solution

#### Paper implementations
- [Hierarchical attention for sentiment classification with visualization](https://github.com/neuromation/ml-recipe-hier-attention)
- [Pediatric bone age assessment](https://github.com/neuromation/ml-recipe-bone-age)
- [Implementation of paper "Tell Me Where to Look: Guided Attention Inference Network"](https://github.com/ngxbac/GAIN)
- [Implementation of paper "Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks"](https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer)
- [Implementation of paper "Utterance-level Aggregation For Speaker Recognition In The Wild"](https://github.com/ptJexio/Speaker-Recognition)
- [Implementation of paper "Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation"](https://github.com/vitrioil/Speech-Separation)

#### Tools and pipelines
- [Catalyst.RL](https://github.com/Scitator/catalyst-rl-framework) – A Distributed Framework for Reproducible RL Research by [Scitator](https://github.com/Scitator)
- [Catalyst.Classification](https://github.com/catalyst-team/classification) - Comprehensive classification pipeline with Pseudo-Labeling by [Bagxi](https://github.com/bagxi) and [Pdanilov](https://github.com/pdanilov)
- [Catalyst.Segmentation](https://github.com/catalyst-team/segmentation) - Segmentation pipelines - binary, semantic and instance, by [Bagxi](https://github.com/bagxi)
- [Catalyst.Detection](https://github.com/catalyst-team/detection) - Anchor-free detection pipeline by [Avi2011class](https://github.com/Avi2011class) and [TezRomacH](https://github.com/TezRomacH)
- [Catalyst.GAN](https://github.com/catalyst-team/gan) - Reproducible GANs pipelines by [Asmekal](https://github.com/asmekal)
- [Catalyst.Neuro](https://github.com/catalyst-team/neuro) - Brain image analysis project, in collaboration with [TReNDS Center](https://trendscenter.org)
- [MLComp](https://github.com/catalyst-team/mlcomp) – distributed DAG framework for machine learning with UI by [Lightforever](https://github.com/lightforever)
- [Pytorch toolbelt](https://github.com/BloodAxe/pytorch-toolbelt) - PyTorch extensions for fast R&D prototyping and Kaggle farming by [BloodAxe](https://github.com/BloodAxe)
- [Helper functions](https://github.com/ternaus/iglovikov_helper_functions) - An unstructured set of helper functions by [Ternaus](https://github.com/ternaus)
- [BERT Distillation with Catalyst](https://github.com/elephantmipt/bert-distillation) by [elephantmipt](https://github.com/elephantmipt)

### Talks
- [Catalyst-team YouTube channel](https://www.youtube.com/channel/UC39Z1Cwr9n8DVpuXcsyi9FQ)
- [Catalyst.RL – reproducible RL research framework](https://docs.google.com/presentation/d/1U6VWIwQnQDGtu6a1x61tt3AlxCJ1-A1EYKd8lR9tKos/edit?usp=sharing) at [Stachka](https://nastachku.ru/archive/2019_innopolis/index.php?dispatch=products.view&product_id=3650)
- [Catalyst.DL – reproducible DL research framework (rus)](https://youtu.be/EfG8iwFNdWg) and [slides (eng)](https://docs.google.com/presentation/d/1TL7N_H31zDFShVbKzLfMC3DYw4e1psj6ScDN8spKQlk/edit?usp=sharing) at [RIF](http://rifvrn.ru/program/catalyst-dl-fast-reproducible-dl-4-html)
- [Catalyst.DL – reproducible DL research framework (rus)](https://youtu.be/7xyMP_5eA8c?t=8964) and [slides (eng)](https://docs.google.com/presentation/d/1XGubfTWvpiJrMyKNx2G6GtAq68y2__sDmx30eSdSRZs/edit?usp=sharing) at [AI-Journey](https://ai-journey.ru/conference-moscow/broadcast?page=2&per-page=12)
- [Catalyst.DL – fast & reproducible DL](https://docs.google.com/presentation/d/1fbF4PMl092kIdjJTw3olR3wI2cl_P2ttN3c9-WTh1gA/edit?usp=sharing) at [Datastart](https://datastart.ru/msk-autumn-2019/)
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://www.youtube.com/watch?v=PprDcJHrFdg&feature=youtu.be&t=4020) and [slides (eng)](https://docs.google.com/presentation/d/1g4g_Rxp9M3xAHwpp_hNzC87L9Gvum3H09g2DIQn1Taw/edit?usp=sharing) at RL reading group Meetup
- [Catalyst – accelerated DL & RL (rus)](https://youtu.be/Rmo2rx5V3v8?t=77) and [slides (eng)](https://docs.google.com/presentation/d/1xMZMjSwJfM5mZMK7pHp6hVI0FxPyZOpRtBZ0J2l1AaY/edit?fbclid=IwAR1q4XJVqYdD-a5oO2n68Y4xHvChIeOSjCSmlUYqrjIzneYpehzF8PiNdMc#slide=id.g75815b5293_0_202) at [Facebook Developer Circle: Moscow | ML & AI Meetup](https://www.facebook.com/groups/475428499888062/)
- [Catalyst.RL - Learn to Move - Walk Around 2nd place solution](https://docs.google.com/presentation/d/14UzYAURBulLjuCbQRnNeROhZ74h51-o460DPTkKMrwo/edit?usp=sharing) at NeurIPS competition track
- [Open Source ML 2019 edition](https://docs.google.com/presentation/d/1A-kwek7USA-j2Nn4n8PmLUQ1PdeUzkkViwXST7RyL-w/edit?usp=sharing) at [Datafest.elka](https://datafest.ru/elka/)



## Community

### Contribution guide

We appreciate all contributions.
If you are planning to contribute back bug-fixes,
please do so without any further discussion.
If you plan to contribute new features, utility functions or extensions,
please first open an issue and discuss the feature with us.

- Please see the [contribution guide](CONTRIBUTING.md) for more information.
- By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).


### User feedback

We have created `catalyst.team.core@gmail.com` for "user feedback".
- If you like the project and want to say thanks, this the right place.
- If you would like to start a collaboration between your team and Catalyst team to do better Deep Learning R&D - you are always welcome.
- If you just don't like Github issues and this ways suits you better - feel free to email us.
- Finally, if you do not like something, please, share it with us and we can see how to improve it.

We appreciate any type of feedback. Thank you!


### Acknowledgments

Since the beginning of the development of the Сatalyst, 
a lot of people have influenced it in a lot of different ways.

#### Catalyst.Team
- [Eugene Kachan](https://www.linkedin.com/in/yauheni-kachan/) ([bagxi](https://github.com/bagxi)) - Config API improvements and CV pipelines
- [Dmytro Doroshenko](https://www.linkedin.com/in/dmytro-doroshenko-05671112a/) ([ditwoo](https://github.com/Ditwoo)) - best ever test cases 
- [Artem Zolkin](https://www.linkedin.com/in/artem-zolkin-b5155571/) ([arquestro](https://github.com/Arquestro)) - documentation grandmaster
- [David Kuryakin](https://www.linkedin.com/in/dkuryakin/) ([dkuryakin](https://github.com/dkuryakin)) - Reaction design

#### Catalyst - Metric Learning team
- [Aleksey Shabanov](https://linkedin.com/in/aleksey-shabanov-96b351189) ([AlekseySh](https://github.com/AlekseySh))
- [Nikita Balagansky](https://www.linkedin.com/in/nikita-balagansky-50414a19a/) ([elephantmipt](https://github.com/elephantmipt))
- [Julia Shenshina](https://github.com/julia-shenshina) ([julia-shenshina](https://github.com/julia-shenshina))

#### Catalyst.Contributors
- [Evgeny Semyonov](https://www.linkedin.com/in/ewan-semyonov/) ([lightforever](https://github.com/lightforever)) - MLComp creator
- [Andrey Zharkov](https://www.linkedin.com/in/andrey-zharkov-8554a1153/) ([asmekal](https://github.com/asmekal)) - Catalyst.GAN initiative
- [Aleksey Grinchuk](https://www.facebook.com/grinchuk.alexey) ([alexgrinch](https://github.com/AlexGrinch)) and [Valentin Khrulkov](https://www.linkedin.com/in/vkhrulkov/) ([khrulkovv](https://github.com/KhrulkovV)) - many RL collaborations
- [Alex Gaziev](https://www.linkedin.com/in/alexgaziev/) ([gazay](https://github.com/gazay)) - a bunch of Config API improvements and our Config API wizard support
- [Eugene Khvedchenya](https://www.linkedin.com/in/cvtalks/) ([bloodaxe](https://github.com/BloodAxe)) - Pytorch-toolbelt library maintainer
- [Yury Kashnitsky](https://www.linkedin.com/in/kashnitskiy/) ([yorko](https://github.com/Yorko)) - Catalyst.NLP initiative

#### Catalyst.Friends
- [Vladimir Iglovikov](https://www.linkedin.com/in/iglovikov/) ([ternaus](https://github.com/ternaus)) - kaggle grandmaster advices
- [Nguyen Xuan Bac](https://www.linkedin.com/in/bac-nguyen-xuan-70340b66/) ([ngxbac](https://github.com/ngxbac)) - kaggle competitions support
- [Ivan Stepanenko](https://www.facebook.com/istepanenko) - awesome Catalyst.Ecosystem design


### Trusted by
- [Awecom](https://www.awecom.com)
- Researchers@[Center for Translational Research in Neuroimaging and Data Science (TReNDS)](https://trendscenter.org)
- [Deep Learning School](https://en.dlschool.org)
- Researchers@[Emory University](https://www.emory.edu)
- [Evil Martians](https://evilmartians.com)
- Researchers@[Georgia Institute of Technology](https://www.gatech.edu)
- Researchers@[Georgia State University](https://www.gsu.edu)
- [Helios](http://helios.to)
- [HPCD Lab](https://www.hpcdlab.com)
- [iFarm](https://ifarmproject.com)
- [Kinoplan](http://kinoplan.io/)
- Researchers@[Moscow Institute of Physics and Technology](https://mipt.ru/english/)
- [Neuromation](https://neuromation.io)
- [Poteha Labs](https://potehalabs.com/en/)
- [Provectus](https://provectus.com)
- Researchers@[Skolkovo Institute of Science and Technology](https://www.skoltech.ru/en)
- [SoftConstruct](https://www.softconstruct.io/)
- Researchers@[Tinkoff](https://www.tinkoff.ru/eng/)
- Researchers@[Yandex.Research](https://research.yandex.com)


### Supported by
- [HostKey](https://www.hostkey.com)
- [Moscow Institute of Physics and Technology](https://mipt.ru/english/)


### Citation

Please use this bibtex if you want to cite this repository in your publications:

    @misc{catalyst,
        author = {Kolesnikov, Sergey},
        title = {Accelerated deep learning R&D},
        year = {2018},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/catalyst-team/catalyst}},
    }
