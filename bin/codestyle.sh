#!/usr/bin/env bash

flake8 . --count --ignore=E126,E226,E704,E731,W503,W504 \
    --max-complexity=16 \
    --inline-quotes "double" \
    --multiline-quotes "double" \
    --docstring-quotes "double" \
    --show-source --statistics
#flake8 . --count --exit-zero --max-complexity=10 --statistics

bash ./yapf.sh --all-in-place
