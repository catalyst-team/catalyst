#!/usr/bin/env bash

set -e

DATAPATH_RAW=""
DATAPATH_PROCESSED=""
BASELOGDIR=""
N_TRIALS=10
THRESHOLD=0.95


# bash argparse
while (( "$#" )); do
  case "$1" in
    --data-raw)
      DATAPATH_RAW=$2
      shift 2
      ;;
    --data-clean)
      DATAPATH_PROCESSED=$2
      shift 2
      ;;
    --baselogdir)
      BASELOGDIR=$2
      shift 2
      ;;
    --n-trials)
      N_TRIALS=$2
      shift 2
      ;;
    --threshold)
      THRESHOLD=$2
      shift 2
      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done

N_CLASS=$(find "${DATAPATH_PROCESSED}" -type d -maxdepth 1 | wc -l | awk '{print $1}')
N_CLASS="$(($N_CLASS-1))"
echo "NUM OF CLASSES: $N_CLASS"

catalyst-data tag2label \
    --in-dir="${DATAPATH_RAW}" \
    --out-dataset="${DATAPATH_RAW}"/dataset.csv \
    --out-labeling="${DATAPATH_RAW}"/tag2cls.json

for ((i=0; i < N_TRIALS; ++i)); do
    LOGDIR="${BASELOGDIR}_${i}"

    catalyst-data tag2label \
        --in-dir="${DATAPATH_PROCESSED}" \
        --out-dataset="${DATAPATH_PROCESSED}"/dataset.csv \
        --out-labeling="${DATAPATH_PROCESSED}"/tag2cls.json

    catalyst-dl train \
        --expdir=finetune \
        --config=autolabel/train.yml \
        --logdir="${LOGDIR}" \
        --stages/data_params/datapath="${DATAPATH_PROCESSED}":str \
        --stages/data_params/in_csv="${DATAPATH_PROCESSED}"/dataset.csv:str \
        --stages/data_params/tag2class="${DATAPATH_PROCESSED}"/tag2cls.json:str \
        --model_params/head_params/n_cls="${N_CLASS}:int"

    catalyst-dl infer \
       --expdir=finetune \
       --resume="${LOGDIR}"/checkpoint.best.pth.tar \
       --out-prefix="${LOGDIR}"/dataset.predictions.{suffix}.npy \
       --config="${LOGDIR}"/config.json,./autolabel/infer.yml \
       --data_params/datapath="${DATAPATH_RAW}":str \
       --data_params/in_csv_infer="${DATAPATH_RAW}"/dataset.csv:str \
       --verbose

    PYTHONPATH=. python ./autolabel/predictions2labels.py \
        --in-npy="${LOGDIR}"/dataset.predictions.infer.logits.npy \
        --in-csv-infer="${DATAPATH_RAW}"/dataset.csv \
        --in-csv-train="${DATAPATH_PROCESSED}"/dataset.csv \
        --in-tag2cls="${DATAPATH_PROCESSED}"/tag2cls.json \
        --in-dir="${DATAPATH_RAW}" \
        --out-dir="${DATAPATH_PROCESSED}"/ \
        --threshold="${THRESHOLD}"
done
