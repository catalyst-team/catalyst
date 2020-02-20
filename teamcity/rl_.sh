#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


echo "apt update && apt install -y redis-server"
apt update && apt install -y redis-server

echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt
