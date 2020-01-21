#!/usr/bin/env bash
pip install tifffile #TODO: check if really required

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

#mkdir -p data
#gdrive_download 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
#tar -xf ./data/isbi.tar.gz -C ./data/

# @TODO: fix macos fail with sed
set -e

# imports check
(set -e; for f in examples/_tests_scripts/dl_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)
(set -e; for f in examples/_tests_scripts/z_*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)


# pipeline 1
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

python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.0
"""

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/trace.py \
  ${LOGDIR}

rm -rf $LOGDIR


# pipeline 2
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

python -c """
from safitty import Safict
metrics = Safict.load('$LOGFILE')
assert metrics.get('stage1.3', 'loss') < metrics.get('stage1.1', 'loss')
assert metrics.get('stage1.3', 'loss') < 2.0
"""


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --resume=${LOGDIR}/checkpoints/best.pth \
  --out_dir=${LOGDIR}/:str \
  --out_prefix="/predictions/":str

python -c """
import numpy as np
data = np.load('${LOGDIR}/predictions/infer.logits.npy')
assert data.shape == (10000, 10)
"""

rm -rf $LOGDIR


# pipeline 3
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 4
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 5
EXPDIR=./examples/_tests_mnist_stages
LOGDIR=./examples/logs/_tests_mnist_stages1
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config6.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 6
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


# pipeline 7
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 8
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config2.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 9
EXPDIR=./examples/_tests_mnist_stages2
LOGDIR=./examples/logs/_tests_mnist_stages2
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR} \
  --check

rm -rf ${LOGDIR}


# pipeline 10
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


# pipeline 11
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
