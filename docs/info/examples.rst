Examples
=================

Run all examples from this dir.

--------------

DL notebooks

1. `features – classification`_

   -  cifar10 classification model
   -  Runner usage example

#. `features – segmentation`_

   -  segmentation with unet
   -  model training and inference
   -  predictions visialization

#. `tutorial – classification`_

    - dataset preparation (raw images -> train/valid/infer splits)
    - augmentations usage example
    - pretrained model finetuning
    - various classification metrics
    - metrics visualizaiton
    - FocalLoss and OneCycle usage examples
    - class imbalance handling
    - model inference


--------------


DL pipelines

1. `features – model training`_

    - configuration files usage example
    - local and docker runs
    - metrics visualization with tensorboard

#. `features – model training with stages`_

    - pipeline example with stages

#. `tutorial – classification pipeline`_

    - classification model training and inference
    - different augmentations and stages usage
    - knn index model example
    - embeddings projector
    - LrFinder usage
    - grid search metrics visualization

#. `tutorial – autolabel`_ - WIP

    - pseudolabeling for your dataset

#. tutorial – segmentation - WIP

#. tutorial – autounet - WIP


--------------


RL pipelines

1. `features – OpenAI Gym LunarLander`_

    - off-policy RL for continuous action space environment
    - DDPG, SAC, TD3 benchmark
    - async multi-cpu, multi-gpu training


#. `features – Atari`_

    - off-policy RL for discrete action space environment
    - DQN
    - image-based environment with various wrappers
    - CNN-based agent with different distribution heads support


--------------


CI tests

1. DL – Mnist with stages
2. RL – OpenAI Gym MountainCarContinuous

.. _features – classification: https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/classification-example.ipynb
.. _features – segmentation: https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-example.ipynb
.. _tutorial – classification: https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
.. _features – model training: https://github.com/catalyst-team/catalyst/tree/master/examples/cifar_simple
.. _features – model training with stages: https://github.com/catalyst-team/catalyst/tree/master/examples/cifar_stages
.. _tutorial – classification pipeline: https://github.com/catalyst-team/classification
.. _tutorial – autolabel: https://github.com/catalyst-team/classification#2-autolabel
.. _features – OpenAI Gym LunarLander: https://github.com/catalyst-team/catalyst/tree/master/examples/rl_gym
.. _features – Atari: https://github.com/catalyst-team/catalyst/tree/master/examples/atari