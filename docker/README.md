## How to build


### CPU only

```bash
docker build -t pro-cpu -f ./Dockerfile.cpu .
```

### GPU

```bash
docker build -t pro-gpu -f ./Dockerfile.gpu .
```


### GPU + extentions

```bash
docker build -t pro-ext -f ./Dockerfile.ext .
```


## How to use

```bash
export LOGDIR=...
docker run -it --rm \
   -v $(pwd):/src -v $LOGDIR:/logdir/ \
   -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=. \
   pro-gpu python prometheus/dl/scripts/train.py \
   --config=./configs/train.yml --logdir=/logdir
```
