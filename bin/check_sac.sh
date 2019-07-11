#!/usr/bin/env bash

redis-server --port 12000 &
sleep 5

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/load_redis.py \
    --in-pkl=./db.dump3.pkl
sleep 5

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_sac_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_sac_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_sac_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_quantile &
sleep 10

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_sac_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_sac_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_sac_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_sac_quantile &
sleep 300

#OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
#    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
#    python catalyst/rl/scripts/dump_redis.py \
#    --out-pkl=./db.dump3.pkl

killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3
killall -9 redis-server
sleep 3

python -c """
import pathlib
import numpy as np
from catalyst import utils

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_sac_base/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print(np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > -5.0
"""

python -c """
import pathlib
import numpy as np
from catalyst import utils

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_sac_categorical/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print(np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > -5.0
"""

python -c """
import pathlib
import numpy as np
from catalyst import utils

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_sac_quantile/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print(np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > -5.0
"""
