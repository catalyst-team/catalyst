#!/usr/bin/env bash
set -e

echo "Training...1"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose

echo "Training...2"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalAvgPool2d:str \
    --model_params/head_params/hiddens=[512]:list

echo "Training...3"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalMaxPool2d:str \
    --model_params/head_params/hiddens=[512]:list

echo "Training...4"
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/head_params/emb_size=128:int

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${BASELOGDIR}
fi
