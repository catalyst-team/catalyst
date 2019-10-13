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

1. [NeurIPS 2018: AI for Prosthetics Challenge](https://github.com/Scitator/neurips-18-prosthetics-challenge)
    - 3rd place solution
2. [Catalyst.RL: A Distributed Framework for Reproducible RL Research](https://github.com/catalyst-team/catalyst-rl-framework)
    - code for paper
3. [NeurIPS 2019: Learn to Move - Walk Around](https://github.com/Scitator/learning-to-move-starter-kit)
    - starter kit
4. [NeurIPS 2019: Animal-AI Olympics](https://github.com/Scitator/animal-olympics-starter-kit)
    - starter kit
5. [ID R&D Anti-spoofing Challenge](https://github.com/bagxi/idrnd-anti-spoofing-challenge-solution)
    - 14th place solution 
6. [NeurIPS 2019: Recursion Cellular Image Classification](https://github.com/ngxbac/Kaggle-Recursion-Cellular)
    - 4th place solution
    - [writeup](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110337#latest-634988)
7. [MICCAI 2019: Automatic Structure Segmentation for Radiotherapy Planning Challenge 2019](https://github.com/ngxbac/StructSeg2019) 
    - 3rd place solution of Task 3: Organ-at-risk segmentation from chest CT scans  
    - 4th place solution of Task 4: Gross Target Volume segmentation of lung cancer
8. [APTOS 2019 Blindness Detection](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
    - 7th place solution
