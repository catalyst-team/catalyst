#!/usr/bin/env bash

redis-server --port 12000 &
sleep 3

wget https://www.dropbox.com/s/5ryokxeksvmbu1u/db.dump.pointenv.pkl

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/load_redis.py \
    --in-pkl=./db.dump.pointenv.pkl

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ddpg_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ddpg_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ddpg_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_quantile &
sleep 600


OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ddpg_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ddpg_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ddpg_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_ddpg_quantile &
sleep 600


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
reward_goal = -8.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ddpg_base/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ddpg_categorical/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ddpg_quantile/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal
"""
