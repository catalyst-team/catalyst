#!/usr/bin/env bash

set -o errexit # Exit on any error

flake8 . \
  --count --ignore=E126,E226,E704,E731,W503,W504 --max-complexity=16 \
  --show-source --statistics
# exit-zero treats all errors as warnings.
flake8 . --count --exit-zero --max-complexity=10 --statistics
# test to make sure the code is yapf compliant
./yapf.sh --all
