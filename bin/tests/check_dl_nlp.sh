#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


echo "check distilbert_text_classification"
LOGFILE=./tests/logs/_tests_nlp_classification/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/distilbert_text_classification \
  --config=./tests/_tests_nlp_classification/config1_basic.yml \
  --logdir=./tests/logs/_tests_nlp_classification \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo "check distilbert_text_classification"
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
assert metrics['train_val.2']['loss'] <= metrics['train_val.1']['loss']
assert metrics['train_val.2']['loss'] < 2.0
"""
rm -rf ./tests/logs/_tests_nlp_classification

echo "train small_max_seq_length"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/distilbert_text_classification \
  --config=./tests/_tests_nlp_classification/config2_small_max_seq_length.yml \
  --logdir=./tests/logs/_tests_nlp_classification \
  --check
rm -rf ./tests/logs/_tests_nlp_classification