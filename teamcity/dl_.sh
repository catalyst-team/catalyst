#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  PYTHON  ##################################

pip install -r requirements/requirements.txt

pip install -r requirements/requirements-cv.txt

pip install -r requirements/requirements-nlp.txt

pip install -r requirements/requirements-rl.txt

pip install alchemy-catalyst


###################################  ENV  #####################################
OMP_NUM_THREADS="1"
MKL_NUM_THREADS="1"


###################################  DATA  ####################################
rm -rf ./data

# load the data
mkdir -p data

if [ -d /mount/isbi ] && [ -d /mount/segmentation_data ]; then
  cp -a /mount/isbi data/isbi
  cp -a /mount/segmentation_data data/segmentation_data
else
  bash bin/scripts/download-gdrive 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
  tar -xf ./data/isbi.tar.gz -C ./data/

  bash bin/scripts/download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj ./data/segmentation_data.zip
  unzip -qqo ./data/segmentation_data.zip -d ./data 2> /dev/null || true
fi


