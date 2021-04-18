## Catalyst: MNIST with stages example

### Local run

From the `examples` folder:

```bash
# 3 options available
bash mnist_stages/run_config.sh  # for Config API run
bash mnist_stages/run_hydra.sh   # for Hydra API run
bash mnist_stages/run_tune.sh    # for Tune run 
```


### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```
