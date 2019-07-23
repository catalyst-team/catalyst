#!/usr/bin/env bash

redis-server --port 12000 &
sleep 3

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_reinforce_discrete.yml \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_discrete &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_reinforce_discrete.yml \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_discrete &
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

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_reinforce_discrete/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal
"""


redis-server --port 12000 &
sleep 3

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_reinforce_continuous.yml \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_continuous &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_reinforce_continuous.yml \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_continuous &
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
reward_goal = -3.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_reinforce_continuous/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > reward_goal
"""
