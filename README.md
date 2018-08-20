# PyTorch common
Common utils for PyTorch DL/RL research. In development. py36.

## Why?

~~just because I can~~

Main reasons:
 - PyTorch is great, but "new model -> new train loop" too much code, as for me.
 - It"s possible to use one common pipeline to train 99.99% neural net models.
 - And do it in reproducible, flexible and structured way.
 - Okay, I just think so.
 - But when it comes to usual PyTorch repos...complicated feelings.
 

Solutions:
 - Key-values storages, key-values storages everywhere!
 - Data and model usage standardization. (okay, v0.5 standardization)
 - Configuration files and loggers (even source code will be saved to logs).
 - Lots of reusable code for different purposes.


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
