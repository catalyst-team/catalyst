## Catalyst Docker


### Base version

```bash
# PyTorch 0.4.1 version
docker build -t catalyst-base -f ./Dockerfile-041 .

# PyTorch 1.0.0 version
docker build -t catalyst-base -f ./Dockerfile-100 .

```

### Contrib version

```bash
# PyTorch 0.4.1 version
docker build -t catalyst-contrib -f ./Dockerfile-contrib-041 .

# PyTorch 1.0.0 version
docker build -t catalyst-contrib -f ./Dockerfile-contrib-100 .
```

## How to use

```bash
export GPUS=...
export LOGDIR=...
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=${GPUS}" \
   catalyst-base catalyst-dl train \
   --config=./configs/train.yml --logdir=/logdir
```
