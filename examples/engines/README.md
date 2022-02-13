# Catalyst engines overview

Let's check different
DataParallel and DistributedDataParallel multi-GPU setups with Catalyst Engines.
PS. multi-node included as well!

> Note: for the Albert training please install requirements with ``pip install datasets transformers``.

### PyTorch
```bash
pip install catalyst
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
CUDA_VISIBLE_DEVICES="0" python train_resnet.py

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=dp

# distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ddp

# multi-node distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ddp \
    --master-addr=127.0.0.1 \
    --world-size=8 \
    --dist-rank=0 \
    --num-workers=8
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0" python train_albert.py

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=dp

# distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ddp

# multi-node distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ddp \
    --master-addr=127.0.0.1 \
    --world-size=8 \
    --dist-rank=0 \
    --num-workers=8
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
CUDA_VISIBLE_DEVICES="0" python train_resnet.py --engine=gpu-amp

CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=dp-amp

# distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ddp-amp

# multi-node distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_resnet.py --engine=ddp-amp \
    --master-addr=127.0.0.1 \
    --world-size=8 \
    --dist-rank=0 \
    --num-workers=8
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

CUDA_VISIBLE_DEVICES="0" python train_albert.py --engine=gpu-amp

CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=dp-amp

# distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ddp-amp

# multi-node distributed training
CUDA_VISIBLE_DEVICES="0,1" python train_albert.py --engine=ddp-amp \
    --master-addr=127.0.0.1 \
    --world-size=8 \
    --dist-rank=0 \
    --num-workers=8
```
</p>
</details>

### PyTorch XLA
```bash
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
pip install catalyst
```

<details open>
<summary>CV - ResNet</summary>
<p>

```bash
python train_resnet.py --engine=e

python train_resnet.py --engine=xla
```
</p>
</details>

<details>
<summary>NLP - Albert</summary>
<p>

```bash
pip install datasets transformers

python train_albert.py --engine=e

python train_albert.py --engine=xla
```
</p>
</details>