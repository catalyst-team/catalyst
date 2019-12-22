#!/usr/bin/env bash
pip install tifffile #TODO: check if really required

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

mkdir -p data
gdrive_download 1N82zh0kzmnzqRvUyMgVOGsCoS1kHf3RP ./data/isbi.tar.gz
tar -xf ./data/isbi.tar.gz -C ./data/

# @TODO: fix macos fail with sed
set -e

(set -e; for f in examples/_tests_scripts/*.py; do PYTHONPATH=./catalyst:${PYTHONPATH} python "$f"; done)

LOGFILE=./examples/logs/_tests_mnist_stages1/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config1.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
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
  ./examples/logs/_tests_mnist_stages1
rm -rf ./examples/logs/_tests_mnist_stages1

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config2.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
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
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config3.yml \
  --resume=./examples/logs/_tests_mnist_stages1/checkpoints/best.pth \
  --out_dir=./examples/logs/_tests_mnist_stages1/:str \
  --out_prefix="/predictions/":str

python -c """
import numpy as np
data = np.load('examples/logs/_tests_mnist_stages1/predictions/infer.logits.npy')
assert data.shape == (10000, 10)
"""
rm -rf ./examples/logs/_tests_mnist_stages1


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config4.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages1


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config5.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages1

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config6.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages1

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config_finder.yml \
  --logdir=./examples/logs/_tests_mnist_stages_finder &
sleep 30
kill %1
rm -rf ./examples/logs/_tests_mnist_stages_finder


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config1.yml \
  --logdir=./examples/logs/_tests_mnist_stages2 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages2


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config2.yml \
  --logdir=./examples/logs/_tests_mnist_stages2 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages2


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config3.yml \
  --logdir=./examples/logs/_tests_mnist_stages2 \
  --check
rm -rf ./examples/logs/_tests_mnist_stages2


PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config_finder.yml \
  --logdir=./examples/logs/_tests_mnist_stages_finder &
sleep 30
kill %1
rm -rf ./examples/logs/_tests_mnist_stages_finder


LOGFILE=./examples/logs/mnist_gan/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/mnist_gan \
  --config=./examples/mnist_gan/config.yml \
  --logdir=./examples/logs/mnist_gan \
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
