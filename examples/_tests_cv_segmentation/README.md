## Test - Catalyst.DL: segmentations with SMP and albumentations

This example is needed for CI test of Catalyst.DL.

### Requirements

Install additional packages: `albumentations` and `smp`

```bash
pip install -U catalyst[cv] # for bash

pip install -U "catalyst[cv]" # for zsh
```

### Get dataset

```bash
mkdir -p data
cd ./_tests_cv_segmentation/data/
download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj segmentation_data.zip
extract-archive segmentation_data.zip
cd ../..
```

### Local run

```bash
catalyst-dl run --configs \
    ./_tests_cv_segmentation/config.yml \
    ./_tests_cv_segmentation/transforms.yml \
    --verbose
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/_tests_cv_segmentation
docker run -it --rm \
   -v $(pwd):/workspace \
   -v $LOGDIR:/logdir/ \
   catalyst-base \
   catalyst-dl run --configs \
    ./_tests_cv_segmentation/config.yml \
    ./_tests_cv_segmentation/transforms.yml \
    --verbose --logdir=/logdir
```
