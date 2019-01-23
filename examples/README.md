# Catalyst examples

Run all examples from this dir.

---

DL notebooks

1. [cifar10 notebook](notebook-example.ipynb)
    - cifar10 classification model
    - Runner usage example
2. [segmentation notebook](segmentation-example.ipynb)
    - segmentation with unet
    - model training and inference
    - predictions visialization

---

DL pipelines
1. [cifar10 model training](cifar_simple)
    - configuration files usage example
    - 
    - local and docker runs
    - tensorboard metrics visualization
2. [cifar10 model training with stages](cifar_stages)
    - pipeline example with stages support
3. [finetune](finetune)
    - classification model training and inference
    - different augmentations and stages usage example
    - index model creating
    - embeddings projector
    - LrFinder usage
    - grid search metrics visualization
4. [autolabel](autolabel)
    - WIP

---

RL pipelines
1. [OpenAI Gym LunarLander](rl_gym)
    - off-policy RL for continuous action space environment
    - DDPG, SAC, TD3 benchmark
    - async multi-cpu, multi-gpu training

---

CI tests

1. DL – Mnist with stages
2. RL – OpenAI Gym MountainCarContinuous
