#!/usr/bin/env bash
set -e

# Parse -s flag which tells us that we should skip inplace yapf
skip_inplace=""
while getopts ":s" flag; do
  case "${flag}" in
    s) skip_inplace="true" ;;
  esac
done


# stop the build if there are any unexpected flake8 issues
bash ./bin/flake8.sh --count \
    --config=./setup.cfg \
    --show-source --statistics

# exit-zero treats all errors as warnings.
flake8 . --count --exit-zero \
    --max-complexity=10 \
    --config=./setup.cfg \
    --statistics

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
    bash ./bin/yapf.sh --all
else
    bash ./bin/yapf.sh --all-in-place
fi

pytest
