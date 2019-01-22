## Catalyst.DL – cifar10 example

### Local run

```bash
catalyst-dl train --config=./cifar_simple/config.yml
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/cifar_simple
docker run -it --rm \
   -v $(pwd):/src -v $LOGDIR:/logdir/ \
   catalyst-image \ 
   catalyst-dl train --config=./cifar_simple/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/cifar_simple
```
