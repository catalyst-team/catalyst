#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh
pip install torch==1.0.0 \
    torchvision==0.2.2 \
    tqdm>=4.33.0 \
    segmentation-models-pytorch==0.0.3
# bash ./teamcity/dl_apex.sh

###################################  CPU ######################################
USE_APEX="0" CUDA_VISIBLE_DEVICES="" bash ./bin/tests/check_dl_all.sh
