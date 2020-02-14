#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  LINUX  ###################################

echo 'apt-get update && apt-get install wget'
apt-get update && apt-get install wget


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


###################################  APEX  ####################################
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex


###################################  CPU ######################################
echo './bin/tests/check_dl_all.sh CPU'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="" \
    bash ./bin/tests/check_dl_all.sh


###################################  GPU ######################################
echo './bin/tests/check_dl_all.sh GPU'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0" \
    bash ./bin/tests/check_dl_all.sh


###################################  GPU2  ####################################
echo './bin/tests/check_dl_all.sh GPU2'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_all.sh

