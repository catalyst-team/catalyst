## Test - Catalyst.DL: segmentations with SMP and albumentations

This example is needed for CI test of Catalyst.DL.

Get dataset

```bash
mkdir -p data
cd ./_test_segmentation/data/
download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj segmentation_data.zip
extract-archive segmentation_data.zip
cd ../..
```

### Local run

```bash
catalyst-dl run --configs \
    ./_test_segmentation/config.yml \
    ./_test_segmentation/transforms.yml \
    --verbose
```

### Docker run

For more information about docker image goto `catalyst/docker`.

```bash
export LOGDIR=$(pwd)/logs/_test_segmentation
docker run -it --rm \
   -v $(pwd):/workspace \
   -v $LOGDIR:/logdir/ \
   catalyst-base \
   catalyst-dl run --configs \
    ./_test_segmentation/config.yml \
    ./_test_segmentation/transforms.yml \
    --verbose --logdir=/logdir
```
