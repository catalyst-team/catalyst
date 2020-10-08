## Config API + Optuna = AutoML

### Local run

```bash
catalyst-dl tune --config=./cifar_stages_optuna/config.yml --verbose
```

### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/cifar_stages_optuna
```
