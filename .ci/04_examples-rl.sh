#!/usr/bin/env bash

set -o errexit # Exit on any error

redis-server --port 12000 &
sleep 10
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/rl/offpolicy/scripts/run_trainer.py \
  --config=./examples/_tests_rl_gym/config.yml \
  --logdir=./examples/logs/_tests_rl_gym \
  --trainer/start_learning=10:int &
 sleep 30
PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/rl/offpolicy/scripts/run_samplers.py \
  --config=./examples/_tests_rl_gym/config.yml \
  --logdir=./examples/logs/_tests_rl_gym \
  --trainer/start_learning=10:int &
sleep 120
kill %3
kill %2
kill %1
rm ./examples/logs/_tests_rl_gym/0.pth
rm ./examples/logs/_tests_rl_gym/1.pth
