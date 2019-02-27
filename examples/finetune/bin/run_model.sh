#!/usr/bin/env bash
set -e

echo "Training..."
catalyst-dl train \
    --expdir=finetune \
    --config=finetune/configs/train_splits.yml \
    --logdir=${LOGDIR} --verbose

echo "Inference..."
catalyst-dl infer \
   --expdir=finetune \
   --resume=${LOGDIR}/checkpoint.best.pth.tar \
   --out-prefix=${LOGDIR}/dataset.predictions.{suffix}.npy \
   --config=${LOGDIR}/config.json,./finetune/configs/infer_splits.yml \
   --verbose

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
