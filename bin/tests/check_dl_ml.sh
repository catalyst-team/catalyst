#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
rm -rf ./tests/logs

################################  pipeline 01  ################################
echo 'pipeline 01'
rm -rf data/split_0
EXPDIR=./tests/_tests_ml_cmcscore
LOGDIR=./tests/logs/_tests_ml_cmcscore
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} \
  --verbose

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 01'
python -c """
from catalyst import utils
import numpy as np
metrics = utils.load_config('$LOGFILE')

EPS = 0.00001
assert np.isclose(metrics['corrupted.1']['cmc_score_1'], 0.1)  # one out of ten is ok
assert np.isclose(metrics['corrupted.1']['cmc_score_1'], 0.1)  # one out of ten is ok
"""

rm -rf {LOGDIR}