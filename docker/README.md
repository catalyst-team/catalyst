## Catalyst Docker


### Base version
Base docker has Catalyst and all needed requirements.
```bash
make docker
```

### Developer version

The developer version contains [packages](/requirements/requirements-dev.txt) for building docs, for checking the code style.
And does not contain Catalyst itself.
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
