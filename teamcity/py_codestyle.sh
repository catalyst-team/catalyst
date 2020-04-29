#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install -r requirements/requirements.txt
pip install -r requirements/requirements-cv.txt
pip install -r requirements/requirements-nlp.txt
pip install -r requirements/requirements-dev.txt
pip install -r docs/requirements.txt

# @TODO: fix server issue
pip install torch==1.4.0 torchvision==0.5.0

#################################  CODESTYLE  #################################

catalyst-check-codestyle
pytest .
make check-docs
