#!/usr/bin/env bash

redis-server --port 12000 &
sleep 3

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_dqn_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_base &
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_dqn_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_categorical &
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_dqn_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_quantile &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_dqn_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_base &
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_dqn_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_categorical &
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_dqn_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_dqn_quantile &
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
reward_goal = 2.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_dqn_base/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_dqn_categorical/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_dqn_quantile/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal
"""
