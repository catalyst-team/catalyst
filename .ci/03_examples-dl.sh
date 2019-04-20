#!/usr/bin/env bash

set -o errexit # Exit on any error
set -o verbose

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config1.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
  --check
python -c "data = open('./examples/logs/_tests_mnist_stages1/metrics.txt', 'r').readlines(); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < float(data[1].rsplit('loss=', 1)[-1][:6]); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < 2.0"
rm -rf ./examples/logs/_tests_mnist_stages1
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config2.yml \
  --logdir=./examples/logs/_tests_mnist_stages1 \
  --check
python -c "data = open('./examples/logs/_tests_mnist_stages1/metrics.txt', 'r').readlines(); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < float(data[1].rsplit('loss=', 1)[-1][:6]); assert float(data[8].rsplit('loss=', 1)[-1][:6]) < 2.0"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config3.yml \
  --resume=./examples/logs/_tests_mnist_stages1/checkpoints/best.pth \
  --out_dir=./examples/logs/_tests_mnist_stages1/:str \
  --out_prefix="/predictions/":str
python -c "import numpy as np; data = np.load('examples/logs/_tests_mnist_stages1/predictions/infer.logits.npy'); assert data.shape == (10000, 10)"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run  \
  --expdir=./examples/_tests_mnist_stages \
  --config=./examples/_tests_mnist_stages/config_finder.yml \
  --logdir=./examples/logs/_tests_mnist_stages_finder &
sleep 30
kill %1
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config1.yml \
  --logdir=./examples/logs/_tests_mnist_stages2 \
  --check
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run  \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config2.yml \
  --logdir=./examples/logs/_tests_mnist_stages2 \
  --check
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -m catalyst.dl run  \
  --expdir=./examples/_tests_mnist_stages2 \
  --config=./examples/_tests_mnist_stages2/config_finder.yml \
  --logdir=./examples/logs/_tests_mnist_stages_finder &
sleep 30
kill %1
