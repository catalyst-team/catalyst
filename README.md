# Prometheus
High-level utils for PyTorch DL/RL research.
It was developed with a focus on reproducibility, fast experimentation and code/ideas/models reusing.
Being able to research/develop something new, rather then write another regular train loop.
Best coding practices included.

## Features

- Universal train/inference loop.
- Key-values storages.
- Data and model usage standardization.
- Configuration files - yaml for model/data hyperparameters.
- Loggers and Tensorboard support.
- Reproducibility - even source code will be saved to logs.
- OneCycle lr scheduler and LRFinder support.
- FP16 support for any model.
- Corrected weight decay (AdamW).
- N-best-checkpoints saving (SWA).
- Training stages support - run whole experiment by one command.
- Logdir autonaming based on hyperparameters - make hyperopt search easy again.
- Callbacks - reusable train/inference pipeline parts (and you can write your own if needed).
- Well structed, so you can just grab a part to your project.
- Lots of reusable code for different purposes: losses, optimizers, models, knns, embeddings projector.

Prometheus is compatible with: Python 3.6. In development.

## How it works

### Deep learning

#### Train
```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. \
    python prometheus/dl/scripts/train.py \
    --config=/path/to/config-file
```

#### Inference
```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. \
    python prometheus/dl/scripts/inference.py \
    --config=/path/to/config-file,/path/to/inference-config-file \
    --resume=/path/to/checkpoint
```


### Reinforcement learning [WIP]

Waiting for NIPS end. Release date - December 2018.

#### Redis server

```bash
redis-server --port 12000
```

#### Trainer

```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. \
    python prometheus/rl/offpolicy/scripts/run_trainer.py \
    --config=/path/to/config-file
```

#### Samplers

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    python prometheus/rl/offpolicy/scripts/run_samplers.py \
    --config=/path/to/config-file
```

## Dependencies
```bash
pip install git+https://github.com/pytorch/tnt.git@master \
    tensorboardX jpeg4py albumentations
```


## Docker

See `./docker` for more information and examples.


## Usage
```bash
git submodule add https://github.com/Scitator/prometheus.git prometheus
```

## Examples

https://github.com/Scitator/prometheus-examples

## Better to use with

CV augmentations – https://github.com/albu/albumentations

## Contribution guide

##### Autoformatting code

We use [yapf](https://github.com/google/yapf) for linting,
and the config file is located at `.style.yapf`.
We recommend running `yapf.sh` prior to pushing to format changed files.


##### Linter

To run the Python linter on a specific file,
run something like `flake8 dl/scripts/train.py`.
You may need to first run `pip install flake8`.

##### Codestyle

tl;dr:
- right margin - 80
- double quotes
- full names: 
    - `model`, `criterion`, `optimizer`, `scheduler` - okay 
    - `mdl`,`crt`, `opt`, `scd` - not okay
- long names solution
    - okay:
    ```bash
    def my_pure_long_name(
            self,
            model, criterion=None, optimizer=None, scheduler=None,
            debug=True):
        """code"""
    ```
    - not okay:
    ```bash
    def my_pure_long_name(self,
                          model, criterion=None, optimizer=None, scheduler=None,
                          debug=True):
        """code"""
    ```
    - why? name refactoring. with first one solution, 
            there are no problems with pep8 codestyle check.
- \* in funcs for force key-value args


## Future features

- prometheus pic
- distributed training (with fp16 support)
- Horovod support
