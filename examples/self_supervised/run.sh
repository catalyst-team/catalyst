export NUM_EPOCHS=1
export BATCH_SIZE=32
export LEARNING_RATE=0.001

for DATASET in "STL10" "CIFAR-10" "CIFAR-100"; do
	for METHOD in "barlow_twins" "byol" "simCLR" "supervised_contrastive"; do
		python $METHOD.py \
			--dataset $DATASET \
			--logdir="./logs/$DATASET/$METHOD" \
			--batch-size=$BATCH_SIZE \
			--epochs=$NUM_EPOCHS \
			--learning-rate=$LEARNING_RATE
	done
done