from typing import Iterable

import torch
from torch import nn


def prepare_optimizable_params(model_or_params, fp16=False):
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()

    master_params = [p for p in params if p.requires_grad]

    if fp16:
        master_params = [
            param.detach().clone().float() for param in master_params
        ]
        for param in master_params:
            param.requires_grad = True

    return master_params


def assert_fp16_available():
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."


__all__ = ["assert_fp16_available", "prepare_optimizable_params"]
