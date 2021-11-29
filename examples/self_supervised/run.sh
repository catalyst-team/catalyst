#!/usr/bin/env bash

# pip install catalyst[cv]==21.11
# pip install catalyst[ml]==21.11

export ARCH="resnet18"
export NUM_EPOCHS=1
export BATCH_SIZE=32
export LEARNING_RATE=0.001

for DATASET in "CIFAR-10" "CIFAR-100" "STL10"; do
	for METHOD in "barlow_twins" "byol" "simCLR" "supervised_contrastive"; do
		python $METHOD.py --dataset $DATASET --verbose \
			--arch=$ARCH \
			--logdir="./logs/$DATASET/$METHOD" \
			--batch-size=$BATCH_SIZE \
			--epochs=$NUM_EPOCHS \
			--learning-rate=$LEARNING_RATE
	done
done
