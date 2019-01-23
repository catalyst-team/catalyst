Examples
=================

Run all examples from this dir.

--------------

DL notebooks

1. `cifar10 notebook`_

   -  cifar10 classification model
   -  Runner usage example

#. `segmentation notebook`_

   -  segmentation with unet
   -  model training and inference
   -  predictions visialization


--------------


DL pipelines

1. `cifar10 model training`_

    - configuration files usage example
    - local and docker runs
    - metrics visualization with tensorboard

#. `cifar10 model training with stages`_

    - pipeline example with stages

#. `finetune`_

    - classification model training and inference
    - different augmentations and stages usage
    - knn index model example
    - embeddings projector
    - LrFinder usage
    - grid search metrics visualization

#. `autolabel`_ - WIP


--------------


RL pipelines

1. `OpenAI Gym LunarLander`_

    - off-policy RL for continuous action space environment
    - DDPG, SAC, TD3 benchmark
    - async multi-cpu, multi-gpu training


--------------


CI tests

1. DL – Mnist with stages
2. RL – OpenAI Gym MountainCarContinuous

.. _cifar10 notebook: https://github.com/catalyst-team/catalyst/blob/master/examples/notebook-example.ipynb
.. _segmentation notebook: https://github.com/catalyst-team/catalyst/blob/master/examples/segmentation-example.ipynb
.. _cifar10 model training: https://github.com/catalyst-team/catalyst/tree/master/examples/cifar_simple
.. _cifar10 model training with stages: https://github.com/catalyst-team/catalyst/tree/master/examples/cifar_stages
.. _finetune: https://github.com/catalyst-team/catalyst/tree/master/examples/finetune
.. _autolabel: https://github.com/catalyst-team/catalyst/blob/master/examples/autolabel
.. _OpenAI Gym LunarLander: https://github.com/catalyst-team/catalyst/tree/master/examples/rl_gym