# Catalyst examples

### DL notebooks

#### Tutorials
1. [tutorial – classification](./notebooks/classification-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb)
    - dataset preparation (raw images -> train/valid/infer splits)
    - augmentations usage example
    - pretrained model finetuning
    - various classification metrics
    - metrics visualizaiton
    - FocalLoss and OneCycle usage examples
    - class imbalance handling
    - model inference
2. [tutorial - segmentation](notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)
    - car segmentation dataset
    - augmentations with [albumentations](https://github.com/albu/albumentations) library
    - training in FP16 with [NVIDIA Apex](https://github.com/NVIDIA/apex)
    - using segmentation models from `catalyst/contrib/models/segmentation`
    - training with multiple criterion (Dice + IoU + BCE) example
    - Lookahead + RAdam optimizer usage example
    - tensorboard logs visualization
    - predictions visualization
    - Test-time augmentations with [ttach](https://github.com/qubvel/ttach) library
3. [tutorial - table data regression](./notebooks/table-data-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/table-data-tutorial.ipynb)
    - dataset California housing dataset(sklearn)
    - StandardScaler preprocessing
    - Simple MLP (40,10,1) linear layers
    - Training + Inference 
    - Results viz.

#### Usage examples
4. [features – classification](./notebooks/notebook-example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/notebook-example.ipynb)
    - cifar10 classification model
    - Runner usage example
5. [features – segmentation](./notebooks/segmentation-example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-example.ipynb)
    - segmentation with unet
    - model training and inference
    - predictions visialization

----

### DL pipelines
1. [features – model training](cifar_simple)
    - configuration files usage example
    - local and docker runs
    - metrics visualization with tensorboard
2. [features – model training with stages](cifar_stages)
    - pipeline example with stages
3. [features - vanilla GAN on MNIST](mnist_gan)
    - experiment with multiple phases & models & optimizers
3. [tutorial – classification](https://github.com/catalyst-team/classification)
    - classification model training and inference
    - different augmentations and stages usage
    - knn index model example
    - embeddings projector
    - LrFinder usage
    - grid search metrics visualization
4. [tutorial – autolabel](https://github.com/catalyst-team/classification#2-autolabel)
    - pseudolabeling for your dataset
5. [tutorial – segmentation][WIP]
6. [tutorial – autounet][WIP]

----

### RL pipelines
1. [features – OpenAI Gym LunarLander](rl_gym)
    - off-policy RL for continuous action space environment
    - DDPG, SAC, TD3 benchmark
    - async multi-cpu, multi-gpu training
2. [features – Atari](atari)
    - off-policy RL for discrete action space environment
    - DQN
    - image-based environment with various wrappers
    - CNN-based agent with different distribution heads support

----

### Catalyst-info
[Link](https://github.com/catalyst-team/catalyst-info)


### Contributions

We supervise the **[Awesome Catalyst list](https://github.com/catalyst-team/awesome-catalyst-list)**. You can make a PR with your project to the list.