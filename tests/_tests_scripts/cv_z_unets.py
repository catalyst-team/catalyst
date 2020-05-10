#!/usr/bin/env python
# coding: utf-8
# flake8: noqa
# isort:skip_file
import os
import sys


if os.getenv("USE_DDP", "0") != "0":
    sys.exit()


# # main check

# In[ ]:

import torch


def check_unet(net_fn):
    net = net_fn()
    # print(net)
    # print("-"*80)
    in_tensor = torch.Tensor(4, 3, 256, 256)
    out_tensor_ = torch.Tensor(4, 1, 256, 256)
    # print(in_tensor.shape)
    # print("-"*80)
    out_tensor = net(in_tensor)
    # print(out_tensor.shape)
    # print("-"*80)
    # print(sum(p.numel() for p in net.parameters()))
    assert out_tensor.shape == out_tensor_.shape, f"{net_fn} feels bad"
    print(f"{net_fn} feels good")
    return net


# ----

# # no pretrain

# In[ ]:

from catalyst.contrib.models.cv import Unet

net = check_unet(Unet)

# In[ ]:

from catalyst.contrib.models.cv import Linknet

net = check_unet(Linknet)

# In[ ]:

from catalyst.contrib.models.cv import FPNUnet

net = check_unet(FPNUnet)

# In[ ]:

from catalyst.contrib.models.cv import PSPnet

net = check_unet(PSPnet)

# ---

# # resnet pretrained

# In[ ]:

from catalyst.contrib.models.cv import ResnetUnet

net = check_unet(ResnetUnet)

# In[ ]:

from catalyst.contrib.models.cv import ResnetLinknet

net = check_unet(ResnetLinknet)

# In[ ]:

from catalyst.contrib.models.cv import ResnetFPNUnet

net = check_unet(ResnetFPNUnet)

# In[ ]:

from catalyst.contrib.models.cv import ResnetPSPnet

net = check_unet(ResnetPSPnet)
