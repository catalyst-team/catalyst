#!/usr/bin/env bash
EXPDIR=cifar_stages
catalyst-dl tune --config=${EXPDIR}/config_tune.yml
