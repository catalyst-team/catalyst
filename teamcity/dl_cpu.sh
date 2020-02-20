#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh


###################################  APEX  ####################################
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex


###################################  CPU ######################################
echo './bin/tests/check_dl_all.sh CPU'
USE_APEX="0" CUDA_VISIBLE_DEVICES="" bash ./bin/tests/check_dl_all.sh
