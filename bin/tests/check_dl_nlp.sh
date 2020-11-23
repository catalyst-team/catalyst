#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


(set -e; for f in tests/_tests_scripts/nlp_*.py; do PYTHONPATH=.:${PYTHONPATH} python "$f"; done)


echo "check distilbert_text_classification"
LOGFILE=./tests/logs/_tests_nlp_classification/checkpoints/_metrics.json

PYTHONPATH=./examples:.:${PYTHONPATH} \
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
PYTHONPATH=./examples:.:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/distilbert_text_classification \
  --config=./tests/_tests_nlp_classification/config2_small_max_seq_length.yml \
  --logdir=./tests/logs/_tests_nlp_classification \
  --check

rm -rf ./examples/logs/_tests_nlp_classification

echo "test text2embedding script"
mkdir ./tmp && \
PYTHONPATH=.:${PYTHONPATH} \
python ./catalyst/contrib/scripts/text2embedding.py \
    --in-csv=examples/distilbert_text_classification/input/train.csv \
    --txt-col="text" \
    --in-huggingface="bert-base-uncased" \
    --out-prefix="./tmp/embeddings" \
    --output-hidden-states \
    --strip \
    --lowercase \
    --remove-punctuation \
    --verbose \
    --batch-size=16 \
    --num-workers=2 \
    --max-length=40 \
    --mask-for-max-length
rm -r ./tmp
