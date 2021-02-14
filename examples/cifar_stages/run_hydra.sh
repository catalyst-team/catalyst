#!/usr/bin/env bash
EXPDIR=cifar_stages
catalyst-dl run --hydra --config-dir ${EXPDIR} --config-name config_hydra.yaml
