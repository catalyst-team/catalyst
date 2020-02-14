#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

echo 'apt-get update && apt-get install wget'
apt-get update && apt-get install wget

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


####################################  DL  ####################################
echo './bin/tests/check_dl.sh'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="" \
    bash ./bin/tests/check_dl.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0" \
    bash ./bin/tests/check_dl.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl.sh


####################################  CV  ####################################
echo './bin/tests/check_dl_cv.sh'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="" \
    bash ./bin/tests/check_dl_cv.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0" \
    bash ./bin/tests/check_dl_cv.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_cv.sh


####################################  GAN  ###################################
echo './bin/tests/check_dl_gan.sh'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="" \
    bash ./bin/tests/check_dl_gan.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0" \
    bash ./bin/tests/check_dl_gan.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0,1" \
    bash ./bin/tests/check_dl_gan.sh


####################################  NLP  ###################################
echo './bin/tests/check_dl_nlp.sh'
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES=""  \
    bash ./bin/tests/check_dl_nlp.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0"  \
    bash ./bin/tests/check_dl_nlp.sh

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" CUDA_VISIBLE_DEVICES="0,1"  \
    bash ./bin/tests/check_dl_nlp.sh
