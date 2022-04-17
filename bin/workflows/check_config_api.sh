#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

DEVICE=${DEVICE:="cpu"}

pip install -e . --quiet --no-deps --upgrade-strategy only-if-needed

################################  pipeline 00  ################################
# checking `catalyst-run` console script entry point

PYTHONPATH="${PYTHONPATH}:." catalyst-run \
  -C "tests/pipelines/configs/test_mnist.yml" \
    "tests/pipelines/configs/engine_${DEVICE}.yml"

rm -rf tests/logs

################################  pipeline 01  ################################
# checking `catalyst-tune` console script entry point

pip install -r requirements/requirements-optuna.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

PYTHONPATH="${PYTHONPATH}:." catalyst-tune \
  -C "tests/contrib/scripts/test_tune.yml" \
    "tests/pipelines/configs/engine_${DEVICE}.yml" \
  --n-trials 2

rm -rf tests/logs
