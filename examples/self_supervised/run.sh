#!/usr/bin/env bash

# pip install catalyst[cv]==21.11
# pip install catalyst[ml]==21.11

export NUM_EPOCHS=1
export BATCH_SIZE=32
export LEARNING_RATE=0.001

for DATASET in "CIFAR-10" "CIFAR-100" "STL10"; do
	for METHOD in "barlow_twins" "byol" "simCLR" "supervised_contrastive"; do
		for ARCH in "resnet18" "resnet34" "resnet50" "resnet101" "resnet152"; do
			python $METHOD.py \
				--dataset $DATASET \
				--arch=$ARCH \
				--logdir="./logs/$DATASET/$METHOD/$ARCH" \
				--batch-size=$BATCH_SIZE \
				--epochs=$NUM_EPOCHS \
				--learning-rate=$LEARNING_RATE
		done
	done
done
