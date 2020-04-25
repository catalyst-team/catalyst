## Catalyst.DL – tiled inference example

Tiled inference is a way to do inference on huge images, which forward pass is
impossible on single GPU due to memory constraints.

### Local run

```bash
catalyst-dl run --config=./tiled_inference/config.yml --verbose
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/tiled_inference
export CUDA_VISIBLE_DEVICES=0
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v ${LOGDIR}:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
   -e "LOGDIR=/logdir" \
   catalyst-base \
   catalyst-dl run --config=./tiled_inference/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
tensorboard --logdir=./logs/tiled_inference
```
