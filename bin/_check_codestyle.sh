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
flake8 . --count --ignore=E126,E226,E704,E731,W503,W504 \
  --max-complexity=16 \
  --inline-quotes "double" \
  --multiline-quotes "double" \
  --docstring-quotes "double" \
  --show-source --statistics

# exit-zero treats all errors as warnings.
flake8 . --count --exit-zero \
  --max-complexity=10 \
  --inline-quotes "double" \
  --multiline-quotes "double" \
  --docstring-quotes "double" \
  --statistics

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
    bash ./bin/yapf.sh --all
else
    bash ./bin/yapf.sh --all-in-place
fi

pytest
