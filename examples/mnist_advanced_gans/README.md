## Catalyst.DL – vanilla GAN on MNIST example

### Local run

#### NOTE: to change runner you should modify \_\_init\_\_.py manually 
```bash
# default GAN
# (Goodfellow et. al., 2014: https://arxiv.org/pdf/1406.2661.pdf)
catalyst-dl run --config=./mnist_advanced_gan/config.yml --verbose

# change runner to WGANRunner
# (Arjovsky et. al., 2017: https://arxiv.org/abs/1701.07875)
catalyst-dl run --config=./mnist_advanced_gan/config_wgan.yml --verbose
# (Gulrahani et. al., 2017: https://arxiv.org/abs/1704.00028)
catalyst-dl run --config=./mnist_advanced_gan/config_wgan_gp.yml --verbose

# change runner to CGanRunner
# (Mirza and Osindero, 2014: https://arxiv.org/abs/1411.1784)
catalyst-dl run --config=./mnist_advanced_gan/config_cgan.yml --verbose
# change runner to ICGanRunner AND experiment to DAGANMnistGanExperiment
# (Antoniou et. al, 2017: https://arxiv.org/pdf/1711.04340.pdf
# note: image conditioning is the same, but training protocol is different)
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
