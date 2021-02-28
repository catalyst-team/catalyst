#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  APEX  ####################################
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex
