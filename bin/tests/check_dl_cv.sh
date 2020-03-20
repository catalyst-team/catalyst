#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
rm -rf ./examples/logs


################################  pipeline 01  ################################
echo 'pipeline 01'
EXPDIR=./examples/_tests_cv_classification
LOGDIR=./examples/logs/_tests_cv_classification
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
echo 'pipeline 01'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.1
"""

echo 'pipeline 01 - trace'
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/trace.py \
  ${LOGDIR}

rm -rf ${LOGDIR}


################################  pipeline 02  ################################
echo 'pipeline 02'
EXPDIR=./examples/_tests_cv_classification
LOGDIR=./examples/logs/_tests_cv_classification
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
echo 'pipeline 02'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.1
"""

if [[ "$USE_DDP" != "1" ]]; then
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
      python catalyst/dl/scripts/run.py \
      --expdir=${EXPDIR} \
      --config=${EXPDIR}/config2_infer.yml \
      --resume=${LOGDIR}/checkpoints/best.pth \
      --out_dir=${LOGDIR}/:str \
      --out_prefix="/predictions/":str

    cat $LOGFILE
    echo 'pipeline 02 - infer'
    python -c """
import numpy as np
data = np.load('${LOGDIR}/predictions/infer.logits.npy')
print(data.shape)
assert data.shape == (10000, 10)
"""

    rm -rf ${LOGDIR}
fi


################################  pipeline 03  ################################
echo 'pipeline 03'
EXPDIR=./examples/_tests_cv_classification
LOGDIR=./examples/logs/_tests_cv_classification
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 03'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.22
"""

rm -rf ${LOGDIR}


################################  pipeline 04  ################################
echo 'pipeline 04'
EXPDIR=./examples/_tests_cv_classification
LOGDIR=./examples/logs/_tests_cv_classification
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 04'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
assert metrics['stage2.2']['loss'] < metrics['stage2.1']['loss']
assert metrics['stage2.2']['loss'] < 2.1
"""

rm -rf ${LOGDIR}


################################  pipeline 05  ################################
echo 'pipeline 05'
EXPDIR=./examples/_tests_cv_classification
LOGDIR=./examples/logs/_tests_cv_classification
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 05'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage2.2']['loss'] < metrics['stage2.1']['loss']
assert metrics['stage2.2']['loss'] < 32.0
"""

rm -rf ${LOGDIR}


################################  pipeline 06  ################################
if [[ "$USE_DDP" != "1" ]]; then
    echo 'pipeline 06 - LrFinder'
    EXPDIR=./examples/_tests_cv_classification
    LOGDIR=./examples/logs/_tests_cv_classification
    LOGFILE=${LOGDIR}/checkpoints/_metrics.json

    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
      python catalyst/dl/scripts/run.py \
      --expdir=${EXPDIR} \
      --config=${EXPDIR}/config6_finder.yml \
      --logdir=${LOGDIR} &

    sleep 30
    kill %1

    if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
        echo "File $LOGFILE does not exist"
        exit 1
    fi

    rm -rf ${LOGDIR}
fi


################################  pipeline 11  ################################
echo 'pipeline 11'
EXPDIR=./examples/_tests_cv_classification_transforms
LOGDIR=./examples/logs/_tests_cv_classification_transforms
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
echo 'pipeline 11'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['best']['loss'] < 2.35
"""

rm -rf ${LOGDIR}


################################  pipeline 12  ################################
echo 'pipeline 12'
EXPDIR=./examples/_tests_cv_classification_transforms
LOGDIR=./examples/logs/_tests_cv_classification_transforms
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
echo 'pipeline 12'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.35
"""

rm -rf ${LOGDIR}


################################  pipeline 13  ################################
echo 'pipeline 13'
EXPDIR=./examples/_tests_cv_classification_transforms
LOGDIR=./examples/logs/_tests_cv_classification_transforms
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 13'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.33
"""

rm -rf ${LOGDIR}


################################  pipeline 14  ################################
if [[ "$USE_DDP" != "1" ]]; then
    echo 'pipeline 14'
    EXPDIR=./examples/_tests_cv_classification_transforms
    LOGDIR=./examples/logs/_tests_cv_classification_transforms
    LOGFILE=${LOGDIR}/checkpoints/_metrics.json

    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
      python catalyst/dl/scripts/run.py \
      --expdir=${EXPDIR} \
      --config=${EXPDIR}/config4_finder.yml \
      --logdir=${LOGDIR} &

    sleep 30
    kill %1

    if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
        echo "File $LOGFILE does not exist"
        exit 1
    fi

    rm -rf ${LOGDIR}
fi


################################  pipeline 15  ################################
echo 'pipeline 15'
EXPDIR=./examples/_tests_cv_classification_transforms
LOGDIR=./examples/logs/_tests_cv_classification_transforms
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5_fp16.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 15'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')
# assert metrics['stage1.2']['loss'] < metrics['stage1.1']['loss']
assert metrics['stage1.2']['loss'] < 2.33
"""

rm -rf ${LOGDIR}


#################################  pipeline 21  ################################
# SEGMENTATION
echo 'pipeline 21 - SEGMENTATION'
EXPDIR=./examples/_tests_cv_segmentation
LOGDIR=./examples/logs/_tests_cv_segmentation
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

## train
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --configs ${EXPDIR}/config.yml ${EXPDIR}/transforms.yml \
  --logdir=${LOGDIR} \
  --stages/data_params/image_path=./data/segmentation_data/train:str \
  --stages/data_params/mask_path=./data/segmentation_data/train_masks:str \
  --check

## check metrics
if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 21 - SEGMENTATION'
python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')

iou = metrics['last']['iou']
loss = metrics['last']['loss']

print('iou', iou)
print('loss', loss)

assert iou > 0.5, f'iou must be > 0.5, got {iou}'
assert loss < 1.4, f'loss must be < 1.4, got {loss}'
"""

## remove logs
rm -rf ./examples/logs/_tests_cv_segmentation


################################  pipeline 99  ################################
rm -rf ./examples/logs
