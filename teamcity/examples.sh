#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

echo 'apt-get update && apt-get install wget'
apt-get update && apt-get install wget

echo 'pip install -r requirements/requirements.txt'
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-cv.txt"
pip install -r requirements/requirements-cv.txt

echo 'pip install -r requirements/requirements-nlp.txt'
pip install -r requirements/requirements-nlp.txt

echo 'pip install -r requirements/requirements-rl.txt'
pip install -r requirements/requirements-rl.txt

echo 'pip install alchemy-catalyst'
pip install alchemy-catalyst

echo './bin/tests/check_dl.sh'
./bin/tests/check_dl.sh

echo './bin/tests/check_cv.sh'
./bin/tests/check_cv.sh

echo './bin/tests/check_nlp.sh'
./bin/tests/check_nlp.sh
