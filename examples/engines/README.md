# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines. 


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
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=apex-dp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=apex-ddp
```

## FairScale
```bash
pip install torch>=1.8.0 catalyst[fairscale]
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-pp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-ddp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-fddp
```

## DeepSpeed
> *Tested under `docker pull deepspeed/deepspeed:v031_torch17_cuda11`.*
```bash
pip install catalyst[deepspeed]
CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=ds-ddp
```
