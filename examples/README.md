# Catalyst examples

## Python API

1. [demo notebook](<./notebooks/demo 21xx.ipynb>) [![Open In Colab](<https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo 21xx.ipynb>)
    - minimal examples
    - Runner customization
    - DL and RL pipelines
1. [classification tutorial](./notebooks/classification-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb)
    - dataset preparation (raw images -> train/valid/infer splits)
    - augmentations usage example
    - pretrained model finetuning
    - various classification metrics
    - metrics visualizaiton
    - FocalLoss and OneCycle usage examples
    - class imbalance handling
    - model inference
1. [segmentation tutorial](notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)
    - car segmentation dataset
    - augmentations with [albumentations](https://github.com/albu/albumentations) library
    - training in FP16 with [NVIDIA Apex](https://github.com/NVIDIA/apex)
    - using segmentation models from `catalyst/contrib/models/segmentation`
    - training with multiple criterion (Dice + IoU + BCE) example
    - Lookahead + RAdam optimizer usage example
    - tensorboard logs visualization
    - predictions visualization
    - Test-time augmentations with [ttach](https://github.com/qubvel/ttach) library
1.  [Pruning tutorial](notebooks/Pruning.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/Pruning.ipynb)
    - Pruning intro
    - Lottery ticket hypothesis
    - Catalyst pruning callback
    - Loading training result from logs

----

## Config API

Full description of configs with comments:
- [Eng](configs/config-description-eng.yml)
- [Rus](configs/config-description-rus.yml)

1. [MNIST with stage](mnist_stages)
    - Config API run example
    - Hydra API run example
    - AutoML Tune example
1. [classification pipeline](https://github.com/catalyst-team/classification)
    - classification model training and inference
    - different augmentations and stages usage
    - metrics visualization with tensorboard
1. [segmentation pipeline](https://github.com/catalyst-team/segmentation)
    - binary and semantic segmentation with U-Net
    - model training and inference
    - different augmentations and stages usage
    - metrics visualization with tensorboard
