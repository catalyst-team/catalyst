## Catalyst.DL – cifar10 with stages example

### Local run

```bash
catalyst-dl run --config=./cifar_stages/config.yml --verbose
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/cifar_stages
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-base \
   catalyst-dl run --config=./cifar_stages/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/cifar_stages
```
