#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh
pip install torch==1.2.0 torchvision==0.4.0
# bash ./teamcity/dl_apex.sh

###################################  CPU ######################################
USE_APEX="0" CUDA_VISIBLE_DEVICES="" bash ./bin/tests/check_dl_all.sh
