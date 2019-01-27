# Catalyst
[![Build Status](https://travis-ci.com/catalyst-team/catalyst.svg?branch=master)](https://travis-ci.com/catalyst-team/catalyst) 
[![License](https://img.shields.io/github/license/catalyst-team/catalyst.svg)](LICENSE)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)

![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)

High-level utils for PyTorch DL & RL research.
It was developed with a focus on reproducibility, 
fast experimentation and code/ideas reusing.
Being able to research/develop something new, 
rather then write another regular train loop.

Break the cycle - use the Catalyst!

---

Catalyst is compatible with: Python 3.6+. PyTorch 0.4.1+.

API documentation and an overview of the library can be found 
[here](https://catalyst-team.github.io/catalyst/index.html).

In the [examples folder](examples) 
of the repository, you can find advanced tutorials and Catalyst best practices.


## Installation

```bash
pip install catalyst
```


## Overview

#### Features

- Universal train/inference loop;
- Configuration files for model/data hyperparameters;
- Reproducibility – even source code will be saved;
- Training stages support;
- Callbacks – reusable train/inference pipeline parts.


#### Structure

- **DL** – runner for training and inference, 
all of the classic machine learning and computer vision metrics 
and a variety of callbacks for training, validation 
and inference of neural networks.
- **RL** – scalable Reinforcement Learning,
actor-critic off-policy continuous actions space algorithms
and their improvements
with distributed training support.
- **contrib** - additional modules contributed by Catalyst users.
- **data** - useful tools and scripts for data processing.


## Getting started: 30 seconds with Catalyst

```python
import torch
from catalyst.dl.runner import SupervisedModelRunner
from your_experiment import get_loaders, get_model, get_callbacks

# experiment setup
logdir = "./logdir"
n_epochs = 42

# data
loaders = get_loaders()

# model and all training stuff
model = get_model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8])

# callbacks - metrics, loggers, etc
callbacks = get_callbacks()

runner = SupervisedModelRunner(
    model=model, criterion=criterion,
    optimizer=optimizer, scheduler=scheduler)
runner.train(
    loaders=loaders, callbacks=callbacks,
    logdir=logdir, epochs=n_epochs, verbose=True)
```

## Docker

Please see the [docker folder](docker) 
for more information and examples.


## Contribution guide

We appreciate all contributions. 
If you are planning to contribute back bug-fixes, 
please do so without any further discussion. 
If you plan to contribute new features, utility functions or extensions, 
please first open an issue and discuss the feature with us.

Please see the [contribution guide](CONTRIBUTING.md) 
for more information.
