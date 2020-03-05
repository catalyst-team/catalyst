#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

echo 'Error'
exit 125

pip install -r requirements/requirements.txt

pip install -r requirements/requirements-cv.txt

pip install -r requirements/requirements-nlp.txt

pip install -r requirements/requirements-rl.txt

pip install -r requirements/requirements-dev.txt

make check-codestyle
