#!/usr/bin/env bash
set -e

echo "projection creating..."
mkdir -p ${LOGDIR}/projector
catalyst-contrib project-embeddings \
   --in-npy=${LOGDIR}/dataset.predictions.infer.embeddings.npy \
   --in-csv="./data/ants_bees/dataset_train.csv" \
   --out-dir=${LOGDIR}/projector \
   --img-size=64 \
   --img-datapath=./data/ants_bees/ \
   --img-col="filepath" \
   --meta-cols="tag"

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
