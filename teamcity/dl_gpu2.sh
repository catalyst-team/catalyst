#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh


###################################  APEX  ####################################
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex


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
