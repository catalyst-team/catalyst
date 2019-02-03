#!/usr/bin/env bash
set -e

DATAROOT="./data/ants_bees"

echo "index model creating..."
catalyst-contrib create-index-model \
  --in-npy=${LOGDIR}/dataset.predictions.infer.embeddings.npy \
  --n-hidden=16 --knn-metric="l2" \
  --out-npy=${LOGDIR}/dataset.predictions.infer.embeddings.pca.npy \
  --out-pipeline=${LOGDIR}/pipeline.embeddings.pkl \
  --out-knn=${LOGDIR}/knn.embeddings.bin \
  --in-npy-test=${LOGDIR}/dataset.predictions.valid.embeddings.npy \
  --out-npy-test=${LOGDIR}/dataset.predictions.valid.embeddings.pca.npy \

echo "index model testing..."
catalyst-contrib check-index-model \
  --in-csv=${DATAROOT}/dataset_train.csv \
  --in-knn=${LOGDIR}/knn.embeddings.bin \
  --in-csv-test=${DATAROOT}/dataset_valid.csv \
  --in-npy-test=${LOGDIR}/dataset.predictions.valid.embeddings.pca.npy \
  --label-column="class" \
  --knn-metric="l2" --batch-size=64 | tee ${LOGDIR}/index_check.txt

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
