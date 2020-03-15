#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./teamcity/dl_.sh
# bash ./teamcity/dl_apex.sh

###################################  CPU ######################################
USE_APEX="0" CUDA_VISIBLE_DEVICES="" bash ./bin/tests/check_dl_all.sh
