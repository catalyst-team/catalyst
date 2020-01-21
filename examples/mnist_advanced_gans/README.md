## Catalyst.DL – vanilla GAN on MNIST example

### Local run

#### NOTE: to change runner you should modify \_\_init\_\_.py manually 
```bash
# default GAN
catalyst-dl run --config=./mnist_advanced_gan/config.yml --verbose
# change runner to WGANRunner
catalyst-dl run --config=./mnist_advanced_gan/config_wgan.yml --verbose
catalyst-dl run --config=./mnist_advanced_gan/config_wgan_gp.yml --verbose
# change runner to CGanRunner
catalyst-dl run --config=./mnist_advanced_gan/config_cgan.yml --verbose
# change runner to ICGanRunner AND experiment to DAGANMnistGanExperiment
catalyst-dl run --config=./mnist_advanced_gan/config_cigan.yml --verbose
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/mnist_gan
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-base \
   catalyst-dl run --config=./mnist_gan/config.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/mnist_gan
```
