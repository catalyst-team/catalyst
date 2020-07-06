#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  PYTHON  ##################################

pip install -r requirements/requirements.txt
pip install -r requirements/requirements-cv.txt
pip install -r requirements/requirements-nlp.txt
pip install -r tests/requirements.txt

# @TODO: fix server issue
pip install torch==1.4.0 torchvision==0.5.0


###################################  ENV  #####################################
OMP_NUM_THREADS="1"
MKL_NUM_THREADS="1"


###################################  DATA  ####################################
rm -rf ./data

# load the data
mkdir -p data

if [[ -d /mount/isbi ]]; then
  cp -a /mount/isbi data/isbi
else
  bash bin/scripts/download-gdrive 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
  tar -xf ./data/isbi.tar.gz -C ./data/
fi

if [[ -d /mount/segmentation_data ]]; then
  cp -a /mount/segmentation_data data/segmentation_data
else
  bash bin/scripts/download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj ./data/segmentation_data.zip
  unzip -qqo ./data/segmentation_data.zip -d ./data 2> /dev/null || true
fi

if [[ -d /mount/MNIST ]]; then
  cp -a /mount/MNIST data/MNIST
else
  bash bin/scripts/download-gdrive 1D_sz7bQSSBDQKNUMSEniXHc9_jHP4EXO ./data/MNIST.zip
  unzip -qqo ./data/MNIST.zip -d ./data 2> /dev/null || true
fi

mkdir -p ~/.cache/torch/checkpoints

if [[ -f /mount/resnext50_32x4d-7cdf4587.pth ]]; then
  cp -a /mount/resnext50_32x4d-7cdf4587.pth ~/.cache/torch/checkpoints/resnext50_32x4d-7cdf4587.pth
else
  wget https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth -O ~/.cache/torch/checkpoints/resnext50_32x4d-7cdf4587.pth
fi

if [[ -f /mount/resnet18-5c106cde.pth ]]; then
  cp -a /mount/resnet18-5c106cde.pth ~/.cache/torch/checkpoints/resnet18-5c106cde.pth
else
  wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O ~/.cache/torch/checkpoints/resnet18-5c106cde.pth
fi
