# Autolabel example

Pseudo is all you need.

### Preparation

```bash
project/
    data/
        data_raw/
            all/
                ...
        data_clean/
            cls_1/
                ...
            cls_N/
                ...
```


### Model training

```bash
export GPUS=""
CUDA_VISIBLE_DEVICES="${GPUS}" bash ./autolabel/autolabel.sh \
    --data-raw ./data/data_raw/ \
    --data-clean ./data/data_clean/ \
    --baselogdir ./logs/autolabel \
    --n-trials 10 \
    --threshold 0.95
```
