#!/usr/bin/env bash
EXPDIR=mnist_stages
catalyst-dl run --hydra --config-dir ${EXPDIR} --config-name config_hydra.yaml
