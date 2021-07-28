
# Before start

0. Please install `mean-average-precision` python package (`pip install mean-average-precision`)
1. Create or convert existing one training and validation datasets to [COCO format](https://cocodataset.org/#format-data) - you need to have `.json` files with annotations.


# Training detector

```bash
catalyst-dl run \
    --config catalyst/examples/detection/ssd-config.yaml \
    --expdir catalyst/examples/detection \
    --logdir detection-logs
```


PYTHONPATH=catalyst python -m catalyst.dl run \
    --config catalyst/examples/detection/ssd-config.yaml \
    --expdir catalyst/examples/detection \
    --logdir detection-logs


PYTHONPATH=catalyst python -m catalyst.dl run \
    --config detection/ssd-config.yaml \
    --expdir detection \
    --logdir detection-logs

catalyst.dl run \
    --config detection/ssd-config.yaml \
    --expdir detection \
    --logdir detection-logs

