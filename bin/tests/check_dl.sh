#!/usr/bin/env bash
pip install tifffile  #TODO: check if really required

mkdir -p data
# gdrive
# gdrive_download 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
# aws
wget https://catalyst-ai.s3-eu-west-1.amazonaws.com/isbi.tar.gz -O ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/

# @TODO: fix macos fail with sed
set -e

# imports check
(set -e; for f in examples/_tests_scripts/*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
#(set -e; for f in examples/_tests_scripts/dl_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
#(set -e; for f in examples/_tests_scripts/z_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)


################################  pipeline 00  ################################
rm -rf ./examples/logs


################################  pipeline 01  ################################
echo 'pipeline 01'
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
"""

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/trace.py \
  ${LOGDIR}

rm -rf $LOGDIR


################################  pipeline 02  ################################
echo 'pipeline 02'
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
"""


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --resume=${LOGDIR}/checkpoints/best.pth \
  --out_dir=${LOGDIR}/:str \
  --out_prefix="/predictions/":str

cat $LOGFILE
python -c """
import numpy as np
data = np.load('${LOGDIR}/predictions/infer.logits.npy')
assert data.shape == (10000, 10)
"""

rm -rf $LOGDIR


################################  pipeline 03  ################################
echo 'pipeline 03'
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.1
"""

rm -rf ${LOGDIR}


################################  pipeline 04  ################################
echo 'pipeline 04'
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage2.3', 'loss') < metrics.get('stage2.1', 'loss')
assert metrics.get('stage2.3', 'loss') < 2.1
"""

rm -rf ${LOGDIR}


################################  pipeline 05  ################################
echo 'pipeline 05'
# LrFinder
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config6.yml \
  --logdir=${LOGDIR} \
  --check

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage2.3', 'loss') < metrics.get('stage2.1', 'loss')
assert metrics.get('stage2.3', 'loss') < 14.5
"""

rm -rf ${LOGDIR}


################################  pipeline 06  ################################
echo 'pipeline 06'
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages_finder
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config_finder.yml \
  --logdir=${LOGDIR} &

sleep 30
kill %1

rm -rf ${LOGDIR}


################################  pipeline 07  ################################
echo 'pipeline 07'
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
# assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('best', 'loss') < 2.35
"""

rm -rf ${LOGDIR}


################################  pipeline 08  ################################
echo 'pipeline 08'
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.35
"""

rm -rf ${LOGDIR}


################################  pipeline 09  ################################
echo 'pipeline 09'
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
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
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
# assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.33
"""

rm -rf ${LOGDIR}


################################  pipeline 10  ################################
echo 'pipeline 10'
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages_finder
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config_finder.yml \
  --logdir=${LOGDIR} &

sleep 30
kill %1

rm -rf ${LOGDIR}

################################  pipeline 11  ################################
# SEGMENTATION
echo 'pipeline 11 - SEGMENTATION'
EXPDIR=./examples/_test_segmentation
LOGDIR=./examples/logs/_test_segmentation
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

## load the data
# mkdir -p ./examples/_test_segmentation/data
# cd ./examples/_test_segmentation/data/
# download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj segmentation_data.zip
# extract-archive segmentation_data.zip
# cd ../../..

## train
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --configs ${EXPDIR}/config.yml ${EXPDIR}/transforms.yml \
  --logdir=${LOGDIR} \
  --stages/data_params/image_path=./examples/_test_segmentation/data/segmentation_data/train:str \
  --stages/data_params/mask_path=./examples/_test_segmentation/data/segmentation_data/train_masks:str \
  --check

## check metrics
if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')

iou = metrics.get('last', 'iou')
loss = metrics.get('last', 'loss')

print('iou', iou)
print('loss', loss)

assert iou > 0.8, f'iou must be > 0.8, got {iou}'
assert loss < 0.2, f'loss must be < 0.2, got {loss}'
"""

## remove logs
rm -rf ./examples/logs/_test_segmentation

################################  pipeline 12  ################################
# GAN
echo 'pipeline 12 -  GAN'
EXPDIR=./examples/mnist_gan
LOGDIR=./examples/logs/mnist_gan
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config.yml \
  --logdir=${LOGDIR} \
  --stages/state_params/num_epochs=11:int

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
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
