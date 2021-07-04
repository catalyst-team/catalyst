# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines. 
> *Please use `pip install git+https://github.com/catalyst-team/catalyst@master --upgrade` before the `v21.06` release.*


## PyTorch
```bash
pip install catalyst
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=dp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=ddp
```

## PyTorch AMP
```bash
pip install torch>=1.8.0 catalyst
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=amp-dp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=amp-ddp
```

## Nvidia APEX
```bash
pip install catalyst && install-apex
# or git clone https://github.com/NVIDIA/apex && cd apex && pip install -e .
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=apex-dp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=apex-ddp
```

## FairScale
> *Tested under `pip install -U torch==1.8.1 fairscale==0.3.7 catalyst==21.06`*
```bash
pip install torch>=1.8.0 catalyst[fairscale]
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-pp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-ddp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-fddp
```

## DeepSpeed
> *Tested under `docker pull deepspeed/deepspeed:v031_torch17_cuda11 and pip install -U torch==1.7.0 deepspeed==0.4.1 catalyst==21.06`.*
```bash
# docker pull deepspeed/deepspeed:v031_torch17_cuda11
# docker run --rm -it -v $(pwd):/workspace deepspeed/deepspeed:v031_torch17_cuda11 /bin/bash
pip install catalyst[deepspeed]
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=ds-ddp
```
