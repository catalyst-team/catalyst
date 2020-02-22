#!/usr/bin/env bash

mkdir -p builds/

sphinx-build -b html ./docs/ builds/ -W --keep-going
CODE=$?

if [[ -z $REMOVE_BUILDS || $REMOVE_BUILDS -eq 1 ]]; then
    rm -rf builds/
fi

echo "#### CODE: ${CODE} ####"
exit ${CODE}
