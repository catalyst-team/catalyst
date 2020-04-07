#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
pip install tifffile  #TODO: check if really required

rm -rf ./tests/logs


################################  pipeline 01  ################################
# imports check
(set -e; for f in tests/_tests_scripts/*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
# (set -e; for f in tests/_tests_scripts/dl_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
# (set -e; for f in tests/_tests_scripts/z_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)


################################  pipeline 99  ################################
rm -rf ./tests/logs
