# PyTorch common
Common utils for PyTorch DL/RL research. In development. py36.


## Features

- Universal train/inference loop.
- Key-values storage, key-values storage everywhere!
- Data and model usage standardization. 
- Configuration files - yaml for model/data hyperparameters.
- Loggers and Tensorboard support.
- Reproducibility - whole source code will be saved to logs.
- OneCycle lr scheduler.
- FP16 support for any model.
- Corrected weight decay (AdamW).
- N-best-checkpoints saving (SWA).
- Callbacks - reusable train/infer pipeline parts (and you can write your own if needed).
- Well structed, so you can just grab a part to your project.
- Lots of reusable code for different purposes (losses, optimizers, models, knns, embeddings projector).


## How it works?

### Deep learning

#### Train
```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. python commom/dl/scripts/train.py \
    --model-dir=/path/to/model \
    --config=/path/to/config-file \
    --logdir=/path/to/logging-directory
```

#### Inference
```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. python commom/dl/scripts/inference.py \
    --model-dir=/path/to/model \
    --config=/path/to/config-file,/path/to/inference-config-file \
    --resume=/path/to/checkpoint
```


### Reinforcement learning (in development)

#### Redis server

```bash
redis-server --port 12000
```

#### Trainer

```bash
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=. \
    python common/rl/offpolicy/scripts/run_trainer.py \
    --algorithm=/path/to/algo-file \
    --config=/path/to/config-file \
    --logdir=/path/to/logging-directory
```

#### Samplers

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    python common/rl/offpolicy/scripts/run_samplers.py \
    --environment=/path/to/env-file \
    --config=/path/to/config-file \
    --logdir=/path/to/logging-directory
    --infer=1 --train=16
```

## Dependencies
```bash
pip install git+https://github.com/pytorch/tnt.git@master
pip install tensorboardX jpeg4py
```

## Usage
```bash
git submodule add https://github.com/Scitator/pytorch-common.git common
```

## Examples

https://github.com/Scitator/pytorch-common-examples

## Better to use with

CV augmentations – https://github.com/albu/albumentations

## Contribution guide

tltr;
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


## Todo

- docs
- tests
- docker
- examples
- auto codestyle check
- distributed training (with fp16 support)
- Horovod
- stable version
