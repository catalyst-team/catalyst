# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines.

> Note: for the Albert training please install requirements with ``pip install datasets transformers``.

## Core

### PyTorch
```bash
pip install catalyst

# CV - ResNet
CUDA_VISIBLE_DEVICES="0" python train_resnet.py --engine=de
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=dp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py \
    --engine=ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0" python train_albert.py --engine=de
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=dp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py \
    --engine=ddp \
    --master_addr=127.0.0.1 \
    --master_port=2112 \
    --world_size=8 \
    --dist_rank=0 \
    --num_workers=8 \
    --sync-bn
```

### PyTorch AMP
```bash
pip install torch>=1.8.0 catalyst

# CV - ResNet
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=amp-dp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=amp-ddp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=amp-ddp --sync-bn

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=amp-dp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=amp-ddp
```

### PyTorch XLA
```bash
pip install catalyst
pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# CV - ResNet
python train_resnet.py --engine=xla
python train_resnet.py --engine=xla-ddp

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=xla
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=xla-ddp
```

## Extensions

### Nvidia APEX
```bash
pip install catalyst && install-apex
# or git clone https://github.com/NVIDIA/apex && cd apex && pip install -e .

# CV - ResNet
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=apex-dp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=apex-ddp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=apex-ddp --sync-bn

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=apex-dp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=apex-ddp
```

### DeepSpeed
> *Tested under `docker pull deepspeed/deepspeed:v031_torch17_cuda11 and pip install -U torch==1.7.0 deepspeed==0.4.1 catalyst==21.06`.*
```bash
# docker pull deepspeed/deepspeed:v031_torch17_cuda11
# docker run --rm -it -v $(pwd):/workspace deepspeed/deepspeed:v031_torch17_cuda11 /bin/bash
pip install catalyst[deepspeed]

# CV - ResNet
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ds-ddp

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ds-ddp
```

### FairScale
> *Tested under `pip install -U torch==1.8.1 fairscale==0.3.7 catalyst==21.06`*
```bash
pip install torch>=1.8.0 catalyst[fairscale]

# CV - ResNet
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-pp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp --sync-bn
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-ddp-amp --sync-bn
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-fddp
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=fs-fddp --sync-bn

# NLP - Albert
pip install datasets transformers
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-pp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-ddp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=fs-fddp
```
