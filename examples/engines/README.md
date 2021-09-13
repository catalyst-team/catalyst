# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines. 

## Core

### PyTorch
```bash
pip install catalyst

CUDA_VISIBLE_DEVICES="0" python train.py --engine=de
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=dp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=ddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=ddp --sync-bn
```

### PyTorch AMP
```bash
pip install torch>=1.8.0 catalyst

CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-dp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-ddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-ddp --sync-bn
```

### PyTorch XLA
```bash
pip install catalyst
pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

python train.py --engine=xla
python train.py --engine=xla-ddp
```

## Extensions

### Nvidia APEX
```bash
pip install catalyst && install-apex
# or git clone https://github.com/NVIDIA/apex && cd apex && pip install -e .

CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-dp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-ddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-ddp --sync-bn
```

### DeepSpeed
> *Tested under `docker pull deepspeed/deepspeed:v031_torch17_cuda11 and pip install -U torch==1.7.0 deepspeed==0.4.1 catalyst==21.06`.*
```bash
# docker pull deepspeed/deepspeed:v031_torch17_cuda11
# docker run --rm -it -v $(pwd):/workspace deepspeed/deepspeed:v031_torch17_cuda11 /bin/bash
pip install catalyst[deepspeed]

CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=ds-ddp
```

### FairScale
> *Tested under `pip install -U torch==1.8.1 fairscale==0.3.7 catalyst==21.06`*
```bash
pip install torch>=1.8.0 catalyst[fairscale]

CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-pp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp --sync-bn
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp-amp --sync-bn
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-fddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-fddp --sync-bn
```
