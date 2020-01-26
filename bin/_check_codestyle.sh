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

echo "isort -rc --check-only --settings-path ./setup.cfg"
isort -rc --check-only --settings-path ./setup.cfg

echo './bin/flake8.sh'
# stop the build if there are any unexpected flake8 issues
bash ./bin/flake8.sh --count \
    --config=./setup.cfg \
    --show-source --statistics

# exit-zero treats all errors as warnings.
echo '~ ~ ~ ~ ~ ~ ~ flake8 warnings ~ ~ ~ ~ ~ ~ ~' 1>&2
flake8 . --count --exit-zero \
    --max-complexity=10 \
    --config=./setup.cfg \
    --statistics
echo '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~' 1>&2

echo 'yapf.sh'

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
    echo 'yapf.sh --all'
    bash ./bin/yapf.sh --all
else
    echo 'yapf.sh --all-in-place'
    bash ./bin/yapf.sh --all-in-place
fi

echo 'pytest'
pytest ./catalyst
