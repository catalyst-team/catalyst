# Torch
# CUDA_VISIBLE_DEVICES="0" python train.py --engine=de
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=dp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=ddp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=ddp --sync-bn

# AMP
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-dp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-ddp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=amp-ddp --sync-bn

# APEX
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-dp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-ddp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=apex-ddp --sync-bn

# FairScale
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-pp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp --sync-bn
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp-amp
CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-ddp-amp --sync-bn
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-fddp
# CUDA_VISIBLE_DEVICES="0,1" python train.py --engine=fs-fddp --sync-bn
