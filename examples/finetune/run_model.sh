#!/usr/bin/env bash
set -e

echo "Training..."
catalyst-dl train \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --logdir=${LOGDIR} --verbose

echo "Inference..."
catalyst-dl inference \
   --model-dir=finetune \
   --resume=${LOGDIR}/checkpoint.best.pth.tar \
   --out-prefix=${LOGDIR}/dataset.predictions.{suffix}.npy \
   --config=${LOGDIR}/config.json,./finetune/inference.yml \
   --verbose

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
