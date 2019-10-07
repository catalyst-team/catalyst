## Catalyst.DL template

### Local run

```bash
catalyst-dl run --config=./configs/config.yml --verbose
```

### Docker run

Build
```bash
make docker
```

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/
docker run -it --rm \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   catalyst-base \
   catalyst-dl run --config=./configs/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/
```
