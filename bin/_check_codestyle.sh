#!/usr/bin/env bash
set -e

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
echo 'flake8: `bash ./bin/flake8.sh`'
bash ./bin/flake8.sh --count \
    --config=./setup.cfg \
    --show-source \
    --statistics

# exit-zero treats all errors as warnings.
echo 'flake8 (warnings): `flake8`'
flake8 . --count --exit-zero \
    --max-complexity=10 \
    --config=./setup.cfg \
    --statistics

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
    echo 'yapf: `bash ./bin/yapf.sh --all`'
    bash ./bin/yapf.sh --all
else
    echo 'yapf: `bash ./bin/yapf.sh --all-in-place`'
    bash ./bin/yapf.sh --all-in-place
fi

echo 'pytest: `pytest`'
pytest ./catalyst
