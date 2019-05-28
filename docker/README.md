## Catalyst Docker


### Base version

```bash
make docker
```

### Developer version

```bash
make docker-dev
```

## How to use

```bash
export GPUS=...
export LOGDIR=...
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=${GPUS}" \
   catalyst-base catalyst-dl run \
   --config=./configs/train.yml --logdir=/logdir
```
