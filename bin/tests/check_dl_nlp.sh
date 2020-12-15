#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


(set -e; for f in tests/_tests_scripts/nlp_*.py; do PYTHONPATH=.:${PYTHONPATH} python "$f"; done)

echo "test text2embedding script"
mkdir ./tmp && \
PYTHONPATH=.:${PYTHONPATH} \
python ./catalyst/contrib/scripts/text2embedding.py \
    --in-csv=assets/text.csv \
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
