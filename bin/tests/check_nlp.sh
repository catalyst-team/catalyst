#!/usr/bin/env bash

# @TODO: fix macos fail with sed
set -e

echo "check distilbert_text_classification"
LOGFILE=./examples/logs/_tests_nlp_classification/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/distilbert_text_classification \
  --config=./examples/_tests_nlp_classification/config1_basic.yml \
  --logdir=./examples/logs/_tests_nlp_classification \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

echo "check results"
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('train_val.3', 'loss') < metrics.get('train_val.1', 'loss')
assert metrics.get('train_val.3', 'loss') < 2.0
"""
rm -rf ./examples/logs/_tests_nlp_classification

echo "train small_max_seq_length"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/distilbert_text_classification \
  --config=./examples/_tests_nlp_classification/config2_small_max_seq_length.yml \
  --logdir=./examples/logs/_tests_nlp_classification \
  --check
rm -rf ./examples/logs/_tests_nlp_classification