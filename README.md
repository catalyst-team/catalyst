# Catalyst
[![Build Status](https://travis-ci.com/catalyst-team/catalyst.svg?branch=master)](https://travis-ci.com/catalyst-team/catalyst)

![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)

High-level utils for PyTorch DL & RL research.
It was developed with a focus on reproducibility, 
fast experimentation and code/ideas reusing.
Being able to research/develop something new, 
rather then write another regular train loop.

Break the cycle - use the Catalyst!

---

Catalyst is compatible with: Python 3.6+. PyTorch 0.4.1+.

Stable branch - `master`. Development branch - `dev`.

API documentation and an overview of the library can be found 
[here]((https://catalyst-team.github.io/catalyst-alpha/index.html)).

In the [examples folder](https://github.com/catalyst-team/catalyst/tree/master/examples) 
of the repository, you can find advanced tutorials and catalyst best practices.


## Installation

```bash
pip install catalyst-ti
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
all of the off-policy continuous actions space algorithms and their improvements
with distributed training support.
- **contrib** - additional modules contributed by Catalyst users.


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
