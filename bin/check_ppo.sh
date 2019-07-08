#!/usr/bin/env bash

redis-server --port 12000 &
sleep 5

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ppo_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_base &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ppo_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_base &
sleep 300

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

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ppo_base/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print(np.mean(checkpoint['rewards']))
assert np.mean(checkpoint['rewards']) > 3.0
"""
