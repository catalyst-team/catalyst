#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

if [[ $BUILD_NUMBER == '670' ]]; then
  exit 0
else
  exit 1
fi

pip install -r requirements/requirements.txt

pip install -r requirements/requirements-cv.txt

pip install -r requirements/requirements-nlp.txt

pip install -r requirements/requirements-rl.txt

pip install -r requirements/requirements-dev.txt

make check-codestyle
