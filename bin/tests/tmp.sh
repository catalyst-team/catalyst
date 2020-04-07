#!/usr/bin/env bash
echo 'pipeline 16'
EXPDIR=./tests/_tests_cv_classification_experiment_registry/test1
LOGDIR=./tests/logs/_tests_cv_classification_experiment_registry/test1
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 16'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.33
"""

rm -rf ${LOGDIR}


echo 'pipeline 17'
EXPDIR=./tests/_tests_cv_classification_experiment_registry/test2
LOGDIR=./tests/logs/_tests_cv_classification_experiment_registry/test2
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 17'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.33
"""

rm -rf ${LOGDIR}


echo 'pipeline 18'
EXPDIR=./tests/_tests_cv_classification_experiment_registry/test2
LOGDIR=./tests/logs/_tests_cv_classification_experiment_registry/test2
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config2.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 18'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.33
"""

rm -rf ${LOGDIR}