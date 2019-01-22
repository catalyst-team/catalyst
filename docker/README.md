## Catalyst Docker


### PyTorch 0.4.1 version

```bash
docker build -t catalyst-image -f ./Dockerfile-041 .
```

### PyTorch 1.0.0 version

```bash
docker build -t catalyst-image -f ./Dockerfile-100 .
```

## How to use

```bash
export GPUS=...
export LOGDIR=...
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/src -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=${GPUS}" \
   catalyst-image catalyst-dl train \
   --config=./configs/train.yml --logdir=/logdir
```
