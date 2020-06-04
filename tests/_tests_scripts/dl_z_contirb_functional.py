#!/usr/bin/env python
# coding: utf-8
# flake8: noqa
import os
import sys

from catalyst.contrib.models import get_convolution_net, get_linear_net

if os.getenv("USE_APEX", "0") != "0" or os.getenv("USE_DDP", "0") != "0":
    sys.exit()


net = get_linear_net(
    in_features=32,
    features=[128, 64, 64],
    use_bias=[True, False, False],
    normalization=[None, "BatchNorm1d", "LayerNorm"],
    dropout_rate=[None, 0.5, 0.8],
    activation=[None, "ReLU", {"module": "ELU", "alpha": 0.5}],
    residual="soft",
)

print(net)

net = get_convolution_net(
    in_channels=3,
    channels=[128, 64, 64],
    kernel_sizes=[8, 4, 3],
    strides=[4, 2, 1],
    groups=[1, 2, 2],
    use_bias=[True, False, False],
    normalization=[None, "BatchNorm2d", "BatchNorm2d"],
    dropout_rate=[None, 0.5, 0.8],
    activation=[None, "ReLU", {"module": "ELU", "alpha": 0.5}],
    residual="soft",
)

print(net)
