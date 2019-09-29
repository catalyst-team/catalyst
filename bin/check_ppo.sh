#!/usr/bin/env bash
# set -e

redis-server --port 12000 &
sleep 3

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ppo_discrete.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_discrete &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ppo_discrete.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_discrete &
sleep 600

killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/dump_db.py \
    --db="redis" \
    --min-reward=2 \
    --chunk-size=100 \
    --out-pkl="./db.dump.out.{suffix}.pkl"
killall -9 redis-server
sleep 3

python -c """
import pathlib
import numpy as np
from catalyst import utils
reward_goal = 2.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ppo_discrete/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal
"""


redis-server --port 12000 &
sleep 3

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_ppo_continuous.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_continuous &
sleep 10

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_ppo_continuous.yml \
    --logdir=./examples/logs/_tests_rl_gym_ppo_continuous &
sleep 600

killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3

OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/dump_db.py \
    --db="redis" \
    --min-reward=2 \
    --chunk-size=100 \
    --out-pkl="./db.dump.out.{suffix}.pkl"
killall -9 redis-server
sleep 3

python -c """
import pathlib
import numpy as np
from catalyst import utils
reward_goal = -1.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_ppo_continuous/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal
"""
