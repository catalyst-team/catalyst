#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install -r requirements/requirements.txt --quiet --find-links https://download.pytorch.org/whl/cpu/torch_stable.html --upgrade-strategy only-if-needed

################################  pipeline 01  ################################
echo 'pipeline 01'
EXPDIR=./examples/cifar_stages_optuna
BASELOGDIR=./examples/logs/cifar_stages_optuna

PYTHONPATH=./examples:.:${PYTHONPATH} \
  python catalyst/dl/scripts/tune.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config.yml \
  --baselogdir=${BASELOGDIR} \
  --n-trials=3 \
  --timeout=600

if [[ ! -d "$BASELOGDIR" ]]; then
  echo "Directory $BASELOGDIR does not exist"
  exit 1
fi
