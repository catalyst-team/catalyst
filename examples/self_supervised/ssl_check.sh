SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python $SCRIPT_DIR/barlow_twins.py \
    --dataset "CIFAR-10" \
	--logdir="./logs" \
	--batch-size=32 \
	--epochs=1 \
	--learning-rate=0.01 \
	--verbose