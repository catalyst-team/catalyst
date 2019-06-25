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

Catalyst helps you write compact
but full-featured DL & RL pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.

#### Features

- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – even source code will be saved.
- Callbacks – reusable train/inference pipeline parts.
- Training stages support.
- Easy customization.
- PyTorch best practices (SWA, AdamW, 1Cycle, FP16 and more).


#### Structure

-  **DL** – runner for training and inference,
   all of the classic machine learning and computer vision metrics
   and a variety of callbacks for training, validation
   and inference of neural networks.
-  **RL** – scalable Reinforcement Learning,
   on-policy & off-policy algorithms and their improvements
   with distributed training support.
-  **contrib** - additional modules contributed by Catalyst users.
-  **data** - useful tools and scripts for data processing.


## Getting started: 30 seconds with Catalyst

```python
import torch
from catalyst.dl import SupervisedRunner

# experiment setup
logdir = "./logdir"
num_epochs = 42

# data
loaders = {"train": ..., "valid": ...}

# model, criterion, optimizer
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
    verbose=True
)
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


## Citation

Please use this bibtex if you want to cite this repository in your publications:

    @misc{catalyst,
        author = {Kolesnikov, Sergey},
        title = {Reproducible and fast DL & RL.},
        year = {2018},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/catalyst-team/catalyst}},
    }
