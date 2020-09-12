## Catalyst.DL – cifar10 with stages example and Optuna AutoML

### Local run

```bash
catalyst-dl run --config=./cifar_stages_optuna/config.yml --verbose
```

### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/cifar_stages_optuna
```
