#!/usr/bin/env bash

flake8 . --count --ignore=E126,E226,E704,E731,W503,W504 --max-complexity=16 --show-source --statistics

./yapf.sh --all-in-place