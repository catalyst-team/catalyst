#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh


###################################  APEX  ####################################
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex


###################################  GPU ######################################
echo './bin/tests/check_dl_all.sh GPU'
USE_APEX="0" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh
USE_APEX="1" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh
