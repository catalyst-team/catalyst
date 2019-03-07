## Catalyst.DL – cifar10 example

### Local run

```bash
catalyst-dl run --config=./cifar_simple/config.yml
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/cifar_simple
docker run -it --rm \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   catalyst-base \
   catalyst-dl run --config=./cifar_simple/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/cifar_simple
```
