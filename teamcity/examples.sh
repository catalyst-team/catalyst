#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  LINUX  ###################################

echo 'apt-get update && apt-get install wget unzip'
apt-get update && apt-get -y install wget unzip


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


###################################  APEX  ####################################
#git clone https://github.com/NVIDIA/apex apex_last
#pip install -v --no-cache-dir \
#    --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex_last
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex

###################################  DATA  ####################################
rm -rf ./data

# load the data
mkdir -p data
bash bin/scripts/download-gdrive 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/

# mkdir -p data
bash bin/scripts/download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj ./data/segmentation_data.zip
unzip -qqo ./data/segmentation_data.zip -d ./data/ 2> /dev/null || true


###################################  CPU ######################################
echo './bin/tests/check_dl_all.sh CPU'
USE_APEX="0" CUDA_VISIBLE_DEVICES="" bash ./bin/tests/check_dl_all.sh


###################################  GPU ######################################
echo './bin/tests/check_dl_all.sh GPU'
USE_APEX="0" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh
USE_APEX="1" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh


###################################  GPU2  ####################################
echo './bin/tests/check_dl_all.sh GPU2'
USE_APEX="0" USE_DDP="0" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_all.sh
USE_APEX="0" USE_DDP="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_all.sh
USE_APEX="1" USE_DDP="0" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_all.sh
USE_APEX="1" USE_DDP="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_all.sh

###################################  DATA  ####################################
rm -rf ./data
