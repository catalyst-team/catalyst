#!/usr/bin/env bash
# set -e

# REINFORCE - 12010
# PPO       - 12011
# DQN       - 12020
# DDOG      - 12021
# SAC       - 12022
# TD3       - 12023
PORT=12010

echo "start redis"
redis-server --port $PORT &
sleep 3

echo "run trainers"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_reinforce_discrete.yml \
    --db/port="$PORT":str \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_discrete &
sleep 10

echo "run samplers"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_reinforce_discrete.yml \
    --db/port="$PORT":str \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_discrete &
sleep 600

echo "kill python processeses"
killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3

echo "dump Redis Database to file"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/dump_db.py \
    --db="redis" \
    --port=$PORT \
    --chunk-size=100 \
    --out-pkl="./db.dump.out.{suffix}.pkl"

echo "kill reids server"
killall -9 redis-server
sleep 3

echo "check results"
python -c """
import pathlib
import numpy as np
from catalyst import utils
reward_goal = 2.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_reinforce_discrete/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal
"""

echo "start redis"
redis-server --port 12000 &
sleep 3

echo "run trainers"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_trainer.py \
    --config=./examples/_tests_rl_gym/config_reinforce_continuous.yml \
    --db/port="$PORT":str \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_continuous &
sleep 10

echo "run samplers"
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/run_samplers.py \
    --config=./examples/_tests_rl_gym/config_reinforce_continuous.yml \
    --db/port="$PORT":str \
    --logdir=./examples/logs/_tests_rl_gym_reinforce_continuous &
sleep 600

echo "kill python processeses"
killall -9 python
sleep 3
killall -9 catalyst-rl
sleep 3

echo "dump Redis Database to file"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
    python catalyst/rl/scripts/dump_db.py \
    --db="redis" \
    --port=$PORT \
    --chunk-size=100 \
    --out-pkl="./db.dump.out.{suffix}.pkl"
killall -9 redis-server
sleep 3

echo "check results"
python -c """
import pathlib
import numpy as np
from catalyst import utils
reward_goal = -6.0

folder = list(pathlib.Path('./examples/logs/_tests_rl_gym_reinforce_continuous/').glob('sampler.valid*'))[0]
checkpoint = utils.load_checkpoint(f'{folder}/checkpoints/best.pth')
print('mean reward', np.mean(checkpoint['reward']))
assert np.mean(checkpoint['reward']) > reward_goal
"""
