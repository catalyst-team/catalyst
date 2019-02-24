#!/usr/bin/env bash
set -e

echo "Training...0"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[1,2,3,4]:list

echo "Training...1"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,2,3,4]:list

echo "Training...2"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,3,4]:list

echo "Training...3"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,2,4]:list

echo "Training...4"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,2,3]:list

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${BASELOGDIR}
fi
