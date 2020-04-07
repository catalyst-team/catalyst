## Catalyst.DL – several GANs on MNIST data example

### Local run

```bash
# (Goodfellow et. al., 2014: https://arxiv.org/pdf/1406.2661.pdf)
catalyst-dl run --config=./mnist_gans/configs/vanilla_gan.yml --verbose
# (Arjovsky et. al., 2017: https://arxiv.org/abs/1701.07875)
catalyst-dl run --config=./mnist_gans/configs/wasserstein_gan.yml --verbose
# (Gulrahani et. al., 2017: https://arxiv.org/abs/1704.00028)
catalyst-dl run --config=./mnist_gans/configs/wasserstein_gan_gp.yml --verbose
# (Mirza and Osindero, 2014: https://arxiv.org/abs/1411.1784)
catalyst-dl run --config=./mnist_gans/configs/conditional_gan.yml --verbose
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
   catalyst-dl run --config=./mnist_gans/configs/vanilla_gan.yml --logdir=/logdir
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/mnist_gans
```
