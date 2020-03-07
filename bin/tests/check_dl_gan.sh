#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
rm -rf ./examples/logs

################################  pipeline 01  ################################
if [[ "$USE_APEX" == "0" ]]; then
    # GAN
    echo 'pipeline 01 -  GAN'
    EXPDIR=./examples/mnist_gans
    LOGDIR=./examples/logs/mnist_gans
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

print('loss_g', loss_g)
print('loss_d_real', loss_d_real)
print('loss_d_fake', loss_d_fake)
print('loss_d', loss_d)

# assert 0.9 < loss_g < 1.5
# assert 0.3 < loss_d_real < 0.6
# assert 0.28 < loss_d_fake < 0.58
# assert 0.3 < loss_d < 0.6
assert loss_g < 2.7
assert loss_d_real < 1.0
assert loss_d_fake < 1.0
assert loss_d < 1.0
"""

    rm -rf ${LOGDIR}
fi
################################  pipeline 99  ################################
rm -rf ./examples/logs
