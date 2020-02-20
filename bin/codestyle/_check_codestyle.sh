#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# Parse -s flag which tells us that we should skip inplace yapf
echo 'parse -s flag'
skip_inplace=""
while getopts ":s" flag; do
  case "${flag}" in
    s) skip_inplace="true" ;;
  esac
done

echo 'isort: `isort -rc --check-only --settings-path ./setup.cfg`'
isort -rc --check-only --settings-path ./setup.cfg

# stop the build if there are any unexpected flake8 issues
echo 'flake8: `bash ./bin/codestyle/_flake8.sh`'
bash ./bin/codestyle/_flake8.sh --count \
    --config=./setup.cfg \
    --show-source \
    --statistics

# exit-zero treats all errors as warnings.
echo 'flake8 (warnings): `bash ./bin/codestyle/_flake8.sh`'
bash ./bin/codestyle/_flake8.sh --count \
    --config=./setup.cfg \
    --show-source \
    --statistics \
    --exit-zero

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
    echo 'yapf: `bash ./bin/codestyle/_yapf.sh --all`'
    bash ./bin/codestyle/_yapf.sh --all
else
    echo 'yapf: `bash ./bin/codestyle/_yapf.sh --all-in-place`'
    bash ./bin/codestyle/_yapf.sh --all-in-place
fi

echo 'pytest: `pytest`'
pytest .
