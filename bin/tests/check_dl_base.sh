#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
pip install tifffile  #TODO: check if really required

# load the data
mkdir -p data
bash bin/scripts/download-gdrive 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/


################################  pipeline 01  ################################
# imports check
(set -e; for f in examples/_tests_scripts/*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
# (set -e; for f in examples/_tests_scripts/dl_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
# (set -e; for f in examples/_tests_scripts/z_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)


################################  pipeline 99  ################################
rm -rf ./examples/logs
rm -rf ./data