#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  LINUX  ###################################

echo 'apt-get update && apt-get install -y wget unzip'
apt-get update && apt-get install -y wget unzip


###################################  PYTHON  ##################################

echo 'pip install -r requirements/requirements.txt'
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-cv.txt"
pip install -r requirements/requirements-cv.txt

echo 'pip install -r requirements/requirements-nlp.txt'
pip install -r requirements/requirements-nlp.txt

echo 'pip install -r requirements/requirements-rl.txt'
pip install -r requirements/requirements-rl.txt

echo 'pip install alchemy-catalyst'
pip install alchemy-catalyst


###################################  ENV  #####################################
OMP_NUM_THREADS="1"
MKL_NUM_THREADS="1"


###################################  DATA  ####################################
rm -rf ./data

# load the data
mkdir -p data
bash bin/scripts/download-gdrive 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/

bash bin/scripts/download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj ./data/segmentation_data.zip
unzip -qqo ./data/segmentation_data.zip -d ./data 2> /dev/null || true
