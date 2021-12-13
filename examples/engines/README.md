# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines.

> Note: for the Albert training please install requirements with ``pip install datasets transformers``.

## Core

### PyTorch
```bash
pip install catalyst
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0" python train_resnet.py --engine=de

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=dp

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0" python train_albert.py --engine=de

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=dp

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

### PyTorch AMP
```bash
pip install torch>=1.8.0 catalyst
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=amp-dp

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=amp-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=amp-dp

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=amp-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

### PyTorch XLA
```bash
pip install catalyst
pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
python train_resnet.py --engine=xla

python train_resnet.py --engine=xla-ddp
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

python train_albert.py --engine=xla

python train_albert.py --engine=xla-ddp
```
</p>
</details>

## Extensions

### Nvidia APEX
```bash
pip install catalyst && install-apex
# or git clone https://github.com/NVIDIA/apex && cd apex && pip install -e .
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=apex-dp

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=apex-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=apex-dp

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=apex-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

### DeepSpeed
> *Tested under `docker pull deepspeed/deepspeed:v031_torch17_cuda11 and pip install -U torch==1.7.0 deepspeed==0.4.1 catalyst==21.12`.*
```bash
# docker pull deepspeed/deepspeed:v031_torch17_cuda11
# docker run --rm -it -v $(pwd):/workspace deepspeed/deepspeed:v031_torch17_cuda11 /bin/bash
pip install catalyst[deepspeed]
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ds-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ds-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

### FairScale
> *Tested under `pip install -U torch==1.8.1 fairscale==0.3.7 catalyst==21.12`*
```bash
pip install torch>=1.8.0 catalyst[fairscale]
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-pp

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp-amp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-fddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-pp

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-ddp-amp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-fddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```
</p>
</details>