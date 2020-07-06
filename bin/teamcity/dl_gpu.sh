#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


bash ./bin/teamcity/dl_.sh
bash ./bin/teamcity/dl_apex.sh

###################################  GPU ######################################
USE_APEX="0" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh
USE_APEX="1" CUDA_VISIBLE_DEVICES="0" bash ./bin/tests/check_dl_all.sh
