
# Before start

0. Please install `mean-average-precision` python package (`pip install mean-average-precision`)
1. Create or convert existing one training and validation datasets to [COCO format](https://cocodataset.org/#format-data) - you need to have `.json` files with annotations.

In this example will be used dataset from [Kaggle - Fruit Detection](https://www.kaggle.com/andrewmvd/fruit-detection).

To convert it to COCO you need to use `to_coco.py` script. This script requires additional package - [`xmltodict`](https://pypi.org/project/xmltodict/).

You can install required packages using next command:

```bash
pip install -r catalyst/examples/detection/requirements.txt
```

Usage is simple:

```bash
python3 to_coco.py <images directory> <annotations directory> <output json file>
```


## Training Single Shot Detector

```bash
catalyst-dl run \
    --config catalyst/examples/detection/ssd-config.yaml \
    --expdir catalyst/examples/detection \
    --logdir ssd-detection-logs
    --verbose
```

## Training CenterNet

```bash
catalyst-dl run \
    --config catalyst/examples/detection/centernet-config.yaml \
    --expdir catalyst/examples/detection \
    --logdir centernet-detection-logs
    --verbose
```
