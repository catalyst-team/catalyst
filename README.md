# Catalyst
High-level utils for PyTorch DL/RL research.
It was developed with a focus on reproducibility, fast experimentation and code/ideas/models reusing.
Being able to research/develop something new, rather then write another regular train loop.
Best coding practices included.

## Features

- Universal train/inference loop.
- Key-values storages.
- Data and model usage standardization.
- Configuration files for model/data hyperparameters.
- Loggers and Tensorboard support.
- Reproducibility – even source code will be saved.
- 1Cycle and LRFinder support.
- FP16 support.
- Corrected weight decay (AdamW).
- N-best-checkpoints saving (SWA).
- Training stages support.
- Logdir autonaming based on hyperparameters.
- Callbacks – reusable train/inference pipeline parts.
- Well structured and production friendly.
- Lots of reusable code for different purposes: losses, optimizers, models, knns, embeddings projector.


Catalyst is compatible with: Python 3.6+. PyTorch 0.4.1+.

Stable branch - `master`. Development branch - `dev`.

## Usage
```bash
git submodule add https://github.com/Scitator/catalyst.git catalyst
```

## Examples

https://github.com/Scitator/catalyst-examples


## Dependencies
```bash
pip install git+https://github.com/pytorch/tnt.git@master \
    tensorboardX jpeg4py albumentations
```

## Docker

See `./docker` for more information and examples.


## Contribution guide

##### Autoformatting code

We use [yapf](https://github.com/google/yapf) for linting,
and the config file is located at `.style.yapf`.
We recommend running `yapf.sh` prior to pushing to format changed files.


##### Linter

To run the Python linter on a specific file,
run something like `flake8 dl/scripts/train.py`.
You may need to first run `pip install flake8`.

See `codestyle.md` for more information.
