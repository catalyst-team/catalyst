#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
rm -rf ./tests/logs

################################  pipeline 01  ################################
if [[ "$USE_APEX" == "0" ]]; then
    # GAN (simple test)
    echo 'pipeline 01 -  GAN'
    EXPDIR=./examples/mnist_gans
    LOGDIR=./tests/logs/mnist_gans
    LOGFILE=${LOGDIR}/checkpoints/_metrics.json

    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
      python catalyst/dl/scripts/run.py \
      --expdir=${EXPDIR} \
      --config=${EXPDIR}/configs/vanilla_gan.yml \
      --logdir=${LOGDIR} \
      --check

    if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
        echo "File $LOGFILE does not exist"
        exit 1
    fi

    cat $LOGFILE
    echo 'pipeline 01 -  GAN'
    python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')

loss_g = metrics['last']['loss_g']
loss_d_real = metrics['last']['loss_d_real']
loss_d_fake = metrics['last']['loss_d_fake']
loss_d = metrics['last']['loss_d']

assert loss_g < 2.7
assert loss_d_real < 1.0
assert loss_d_fake < 1.0
assert loss_d < 1.0
"""

    rm -rf ${LOGDIR}
fi

################################  pipeline 02  ################################
if [[ "$USE_APEX" == "0" ]]; then
    # conditional Wasserstein GAN-GP (more complicated test)
    echo 'pipeline 02 - conditional WGAN-GP'
    EXPDIR=./examples/mnist_gans
    LOGDIR=./tests/logs/mnist_gans
    LOGFILE=${LOGDIR}/checkpoints/_metrics.json

    CHECK_BATCH_STEPS=30 \
    CHECK_EPOCH_STEPS=10 \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
      python catalyst/dl/scripts/run.py \
      --expdir=${EXPDIR} \
      --config=${EXPDIR}/configs/conditional_wasserstein_gan_gp.yml \
      --logdir=${LOGDIR} \
      --check

    if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
        echo "File $LOGFILE does not exist"
        exit 1
    fi

    cat $LOGFILE
    echo 'pipeline 02 -  conditional WGAN-GP'
    python -c """
from catalyst import utils
metrics = utils.load_config('$LOGFILE')

loss_g = metrics['last']['loss_g']
loss_d_real = metrics['last']['loss_d_real']
loss_d_fake = metrics['last']['loss_d_fake']
loss_d = metrics['last']['loss_d']

w_dist = metrics['last']['wasserstein_distance']

assert -100 < loss_g < 100
assert -100 < loss_d_real < 100
assert -100 < loss_d_fake < 100
assert -100 < loss_d < 100
assert -30 < w_dist < 30
"""

    rm -rf ${LOGDIR}
fi
################################  pipeline 99  ################################
rm -rf ./tests/logs
