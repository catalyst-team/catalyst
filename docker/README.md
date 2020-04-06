## Catalyst Docker
[![Docker Pulls](https://img.shields.io/docker/pulls/catalystteam/catalyst)](https://hub.docker.com/r/catalystteam/catalyst/tags)

- `catalystteam/catalyst:{CATALYST_VERSION}` – simple image with Catalyst
- `catalystteam/catalyst:{CATALYST_VERSION}-fp16` – Catalyst with FP16
- `catalystteam/catalyst:{CATALYST_VERSION}-dev` – Catalyst for development with all the requirements
- `catalystteam/catalyst:{CATALYST_VERSION}-dev-fp16` – Catalyst for development with FP16

### Base version
Base docker has Catalyst and all needed requirements.
```bash
make docker
```

With FP16
```bash
make docker-fp16
```

### Developer version

The developer version contains [packages](/requirements/requirements-dev.txt) for building docs, for checking the code style.
And does not contain Catalyst itself.
```bash
make docker-dev
```

With FP16
```bash
make docker-dev-fp16
```

## How to use

```bash
export GPUS=...
export LOGDIR=...
docker run -it --rm --runtime=nvidia \
   -v $(pwd):/workspace -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=${GPUS}" \
   catalyst-base catalyst-dl run \
   --config=./configs/train.yml --logdir=/logdir
```
