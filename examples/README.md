# Catalyst examples

Run all examples from this dir.

---

DL notebooks

1. [cifar10 notebook](notebook-example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebook-example.ipynb)
    - cifar10 classification model
    - Runner usage example
2. [segmentation notebook](segmentation-example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/segmentation-example.ipynb)
    - segmentation with unet
    - model training and inference
    - predictions visialization

---

DL pipelines
1. [cifar10 model training](cifar_simple)
    - configuration files usage example
    - local and docker runs
    - metrics visualization with tensorboard
2. [cifar10 model training with stages](cifar_stages)
    - pipeline example with stages
3. [classification](https://github.com/catalyst-team/classification)
    - classification model training and inference
    - different augmentations and stages usage
    - knn index model example
    - embeddings projector
    - LrFinder usage
    - grid search metrics visualization
4. [autolabel](https://github.com/catalyst-team/classification#autolabel)
    - pseudolabeling for your dataset

---

RL pipelines
1. [OpenAI Gym LunarLander](rl_gym)
    - off-policy RL for continuous action space environment
    - DDPG, SAC, TD3 benchmark
    - async multi-cpu, multi-gpu training
2. [Atari](atari)
    - off-policy RL for discrete action space environment
    - DQN
    - image-based environment with various wrappers
    - CNN-based agent with different distribution heads support

---

CI tests

1. DL – Mnist with stages
2. RL – OpenAI Gym MountainCarContinuous
