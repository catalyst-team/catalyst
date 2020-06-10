Examples
========


Tutorials
---------

1. `classification tutorial`_
    - dataset preparation (raw images -> train/valid/infer splits)
    - augmentations usage example
    - pretrained model finetuning
    - various classification metrics
    - metrics visualizaiton
    - FocalLoss and OneCycle usage examples
    - class imbalance handling
    - model inference

#. `segmentation tutorial`_
    - car segmentation dataset
    - augmentations with `albumentations`_ library
    - training in FP16 with `NVIDIA Apex`_
    - using segmentation models from ``catalyst/contrib/models/cv/segmentation``
    - training with multiple criterion (Dice + IoU + BCE) example
    - Lookahead + RAdam optimizer usage example
    - tensorboard logs visualization
    - predictions visualization
    - Test-time augmentations with `ttach`_ library


Pipelines
---------

1. Full description of configs with comments:
    - `Eng`_
    - `Rus`_

#. `classification pipeline`_
    - classification model training and inference
    - different augmentations and stages usage
    - metrics visualization with tensorboard
#. `segmentation pipeline`_
    - binary and semantic segmentation with U-Net
    - model training and inference
    - different augmentations and stages usage
    - metrics visualization with tensorboard


RL tutorials & pipelines
------------------------

For Reinforcement Learning examples check out our `Catalyst.RL repo`_.


.. _classification tutorial: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
.. _segmentation tutorial: https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb
.. _classification pipeline: https://github.com/catalyst-team/classification
.. _segmentation pipeline: https://github.com/catalyst-team/segmentation
.. _Eng: https://github.com/catalyst-team/catalyst/blob/master/examples/configs/config-description-eng.yml
.. _Rus: https://github.com/catalyst-team/catalyst/blob/master/examples/configs/config-description-rus.yml
.. _Catalyst.RL repo: https://github.com/catalyst-team/catalyst-rl

.. _albumentations: https://github.com/albu/albumentations
.. _NVIDIA Apex: https://github.com/NVIDIA/apex
.. _ttach: https://github.com/qubvel/ttach
