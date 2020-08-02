#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install \
    -r ./requirements/requirements.txt \
    -r ./requirements/requirements-dev.txt \
    -r ./requirements/requirements-ml.txt \
    -r ./requirements/requirements-cv.txt \
    -r ./requirements/requirements-nlp.txt \
    -r ./requirements/requirements-contrib.txt

# @TODO: fix server issue
pip install torch==1.4.0 torchvision==0.5.0

#################################  CODESTYLE  #################################

catalyst-check-codestyle
pytest .
make check-docs
