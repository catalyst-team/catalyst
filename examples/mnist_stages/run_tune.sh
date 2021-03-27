#!/usr/bin/env bash
EXPDIR=mnist_stages
catalyst-dl tune --config=${EXPDIR}/config_tune.yml --n-trials=3
