# Catalyst examples

## Python API

Catalyst Python API examples can be found in the 
[minimal examples](https://github.com/catalyst-team/catalyst#minimal-examples) 
and [notebook section](https://github.com/catalyst-team/catalyst#notebooks).

### Notebooks

- Introduction tutorial "[Customizing what happens in `train`](./notebooks/customizing_what_happens_in_train.ipynb)" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/customizing_what_happens_in_train.ipynb)
- Demo with [customization examples](./notebooks/customization_tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/customization_tutorial.ipynb)
- [Reinforcement Learning examples with Catalyst](./notebooks/reinforcement_learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/reinforcement_learning.ipynb)

### Scripts

- [Reinforcement Learning with Catalyst](./reinforcement_learning)

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