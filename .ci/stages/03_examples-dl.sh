#!/usr/bin/env bash

set -o errexit # Exit on any error
#set -o verbose
set -x

export LOGDIR=${LOGDIR:-logs/example/}
export PROJECT_ROOT=$(realpath "${0%/*}/../../")
export PYTHONPATH=$PROJECT_ROOT

python3 -m catalyst.dl run \
  --expdir $PROJECT_ROOT/examples/_tests_mnist_stages \
  --config $PROJECT_ROOT/examples/_tests_mnist_stages/config1.yml \
  --logdir $LOGDIR/_tests_mnist_stages1 \
  --check
python3 -c "data = open('$LOGDIR/_tests_mnist_stages1/metrics.txt', 'r').readlines(); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < float(data[1].rsplit('loss=', 1)[-1][:6]); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < 2.0"
rm -rf $LOGDIR/_tests_mnist_stages1

python3 -m catalyst.dl run \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages/config2.yml \
  --logdir=$LOGDIR/_tests_mnist_stages1 \
  --check

python3 -c "data = open('$LOGDIR/_tests_mnist_stages1/metrics.txt', 'r').readlines(); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < float(data[1].rsplit('loss=', 1)[-1][:6]); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < 2.0"

python3 -m catalyst.dl run \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages/config3.yml \
  --resume=$LOGDIR/_tests_mnist_stages1/checkpoints/best.pth \
  --out_dir=$LOGDIR/_tests_mnist_stages1/:str \
  --out_prefix="/predictions/":str

python3 -c "import numpy as np; data = np.load('$LOGDIR/_tests_mnist_stages1/predictions/infer.logits.npy'); assert data.shape == (10000, 10)"

python3 -m catalyst.dl run  \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages/config_finder.yml \
  --logdir=$LOGDIR/_tests_mnist_stages_finder &

sleep 30
kill %1

python3 -m catalyst.dl run \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages2 \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages2/config1.yml \
  --logdir=$LOGDIR/_tests_mnist_stages2 \
  --check

python3 -m catalyst.dl run  \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages2 \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages2/config2.yml \
  --logdir=$LOGDIR/_tests_mnist_stages2 \
  --check

python3 -m catalyst.dl run  \
  --expdir=$PROJECT_ROOT/examples/_tests_mnist_stages2 \
  --config=$PROJECT_ROOT/examples/_tests_mnist_stages2/config_finder.yml \
  --logdir=$LOGDIR/_tests_mnist_stages_finder &

sleep 30
kill %1
