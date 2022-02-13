#!/usr/bin/env bash

# pip install catalyst[cv]==22.02
# pip install catalyst[ml]==22.02

export NUM_EPOCHS=20
export BATCH_SIZE=256
export LEARNING_RATE=0.001

for DATASET in "CIFAR-10" "CIFAR-100" "STL10"; do
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
