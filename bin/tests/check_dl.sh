#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


pip install tifffile  #TODO: check if really required

mkdir -p data
# gdrive
# gdrive_download 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
# aws
wget https://catalyst-ai.s3-eu-west-1.amazonaws.com/isbi.tar.gz -O ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/

# imports check
(set -e; for f in examples/_tests_scripts/*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
#(set -e; for f in examples/_tests_scripts/dl_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
#(set -e; for f in examples/_tests_scripts/z_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
