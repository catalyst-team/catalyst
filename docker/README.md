## How to build


### CPU

```bash
docker build -t catalyst-cpu -f ./Dockerfile.cpu .
```

### GPU

```bash
docker build -t catalyst-gpu -f ./Dockerfile.gpu .
```

## How to use

```bash
export LOGDIR=...
docker run -it --rm \
   -v $(pwd):/src -v $LOGDIR:/logdir/ \
   -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=. \
   catalyst-gpu python catalyst/dl/scripts/train.py \
   --config=./configs/train.yml --logdir=/logdir
```
