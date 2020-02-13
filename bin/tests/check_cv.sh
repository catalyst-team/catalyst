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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
"""

echo 'pipeline 01 - trace'
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/trace.py \
  ${LOGDIR}

rm -rf $LOGDIR


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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
"""


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
assert data.shape == (10000, 10)
"""

rm -rf $LOGDIR


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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage2.3', 'loss') < metrics.get('stage2.1', 'loss')
assert metrics.get('stage2.3', 'loss') < 2.1
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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage2.3', 'loss') < metrics.get('stage2.1', 'loss')
assert metrics.get('stage2.3', 'loss') < 23.0
"""

rm -rf ${LOGDIR}


################################  pipeline 06  ################################
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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
# assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('best', 'loss') < 2.35
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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
# assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.35
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
from safitty import Safict
metrics = Safict.load('$LOGFILE')
# assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.33
"""

rm -rf ${LOGDIR}


################################  pipeline 14  ################################
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

################################  pipeline 21  ################################
# SEGMENTATION
echo 'pipeline 21 - SEGMENTATION'
EXPDIR=./examples/_tests_cv_segmentation
LOGDIR=./examples/logs/_tests_cv_segmentation
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

# load the data
mkdir -p ./examples/_tests_cv_segmentation/data
cd ./examples/_tests_cv_segmentation/data/
download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj segmentation_data.zip
extract-archive segmentation_data.zip
cd ../../..

## train
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --configs ${EXPDIR}/config.yml ${EXPDIR}/transforms.yml \
  --logdir=${LOGDIR} \
  --stages/data_params/image_path=./examples/_tests_cv_segmentation/data/segmentation_data/train:str \
  --stages/data_params/mask_path=./examples/_tests_cv_segmentation/data/segmentation_data/train_masks:str \
  --check

## check metrics
if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 21 - SEGMENTATION'
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')

iou = metrics.get('last', 'iou')
loss = metrics.get('last', 'loss')

print('iou', iou)
print('loss', loss)

assert iou > 0.8, f'iou must be > 0.8, got {iou}'
assert loss < 0.32, f'loss must be < 0.32, got {loss}'
"""

## remove logs
rm -rf ./examples/logs/_tests_cv_segmentation

################################  pipeline 31  ################################
# GAN
echo 'pipeline 31 -  GAN'
EXPDIR=./examples/mnist_gans
LOGDIR=./examples/logs/mnist_gans
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/configs/vanilla_gan.yml \
  --logdir=${LOGDIR} \
  --stages/state_params/num_epochs=11:int

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 31 -  GAN'
python -c """
from safitty import Safict
metrics=Safict.load('$LOGFILE')

loss_g = metrics.get('last', 'loss_g')
loss_d_real = metrics.get('last', 'loss_d_real')
loss_d_fake = metrics.get('last', 'loss_d_fake')
loss_d = metrics.get('last', 'loss_d')

print('loss_g', loss_g)
print('loss_d_real', loss_d_real)
print('loss_d_fake', loss_d_fake)
print('loss_d', loss_d)

# assert 0.9 < loss_g < 1.5
# assert 0.3 < loss_d_real < 0.6
# assert 0.28 < loss_d_fake < 0.58
# assert 0.3 < loss_d < 0.6
assert loss_g < 2.0
assert loss_d_real < 0.9
assert loss_d_fake < 0.9
assert loss_d < 0.9
"""

rm -rf ${LOGDIR}

################################  pipeline 99  ################################
rm -rf ./examples/logs
