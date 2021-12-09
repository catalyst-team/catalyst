#!/usr/bin/env bash

# pip install catalyst[cv]==21.11
# pip install catalyst[ml]==21.11

export NUM_EPOCHS=20
export BATCH_SIZE=256
export LEARNING_RATE=0.001

for DATASET in "MNIST"; do
	for METHOD in "barlow_twins" "byol" "simCLR" "supervised_contrastive"; do
		python $METHOD.py \
			--dataset $DATASET \
			--logdir="./logs/$DATASET/$METHOD" \
			--batch-size=$BATCH_SIZE \
			--epochs=$NUM_EPOCHS \
			--learning-rate=$LEARNING_RATE \
			--verbose
	done
done
