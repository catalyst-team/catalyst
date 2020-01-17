#!/usr/bin/env bash
# set -e

echo "start redis"
redis-server --port 12000 &
sleep 3

echo "download data"
wget https://catalyst-ai.s3-eu-west-1.amazonaws.com/db.dump.pointenv.190821.pkl

echo "load data to Redis Database"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/load_db.py \
    --db="redis" \
    --in-pkl=./db.dump.pointenv.190821.pkl

echo "run trainers"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_td3_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_td3_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_td3_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_quantile &
sleep 900

echo "run samplers"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_td3_base.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_base &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_td3_categorical.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_categorical &
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_td3_quantile.yml \
    --logdir=./examples/logs/_tests_rl_gym_td3_quantile &
sleep 300

echo "kill python processes"
killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3

echo "dump Redis Database to file"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/dump_db.py \
    --db="redis" \
    --out-pkl=./db.dump.pointenv.190821.out.pkl

echo "kill redis server"
killall -9 redis-server
sleep 3

echo "check results"
python -c """
import pathlib
import numpy as np
from catalyst import utils
reward_goal = -8.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_td3_base/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_td3_categorical/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_td3_quantile/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal
"""
