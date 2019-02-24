## Catalyst.DL – resnet finetune example

KNN is all you need.

### Goals

Main
- tune ResnetEncoder
- train MiniNet for image classification
- learn embeddings representation
- create knn index model

Additional
- visualize embeddings with TF.Projector
- find best starting lr with LRFinder
- plot grid search metrics and compare different approaches

### Preparation

Get the [data](https://www.dropbox.com/s/9438wx9ku9ke1pt/ants_bees.tar.gz)
```bash
wget https://www.dropbox.com/s/9438wx9ku9ke1pt/ants_bees.tar.gz
tar -xvf ./ants_bees.tar.gz
```

and unpack it to `catalyst/examples/data` folder:
```bash
catalyst/examples/data/
    ants_bees/
        ants/
            ...
        bees/
            ...
```

Process the data
```bash
catalyst-data tag2label \
    --in-dir=./data/ants_bees \
    --out-dataset=./data/ants_bees/dataset.csv \
    --out-labeling=./data/ants_bees/tag2cls.json

python finetune/prepare_splits.py \
    --in-csv=./data/ants_bees/dataset.csv \
    --tag2class=./data/ants_bees/tag2cls.json \
    --tag-column=tag \
    --class-column=class \
    --n-folds=5 \
    --train-folds=0,1,2,3 \
    --out-csv=./data/ants_bees/dataset_folds.csv \
    --out-csv-train=./data/ants_bees/dataset_train.csv \
    --out-csv-valid=./data/ants_bees/dataset_valid.csv
```

And `pip install tensorflow` for visualization.

### Docker

For more information about docker image goto `catalyst/docker`.

### Model training

```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-base bash finetune/bin/run_model.sh
```

### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/finetune
```

### Embeddings projecting

```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   catalyst-contrib bash finetune/bin/run_projector.sh
tensorboard --logdir=./logs/finetune/projector

```

### Index model training

```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "LOGDIR=/logdir" \
   catalyst-contrib bash finetune/bin/run_index.sh
```

### LrFinder example

```bash
export LOGDIR=$(pwd)/logs/finetune/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-base bash finetune/bin/run_lrfinder.sh
```

### Grid search metrics visualization

```bash
export BASELOGDIR=$(pwd)/logs/finetune
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-base bash finetune/bin/run_grid.sh
```


### KFold training metrics visualization

```bash
export BASELOGDIR=$(pwd)/logs/finetune/kfold
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-base bash finetune/bin/run_kfold.sh
```