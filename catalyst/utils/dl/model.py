from typing import Iterable

import torch
from torch import nn


def get_optimizable_params(model_or_params):
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()

    master_params = [p for p in params if p.requires_grad]
    return master_params


def assert_fp16_available():
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."

    try:
        __import__('apex')
    except ImportError:
        assert False, \
            "NVidia Apex package must be installed. " \
            "See https://github.com/NVIDIA/apex."


__all__ = ["assert_fp16_available", "get_optimizable_params"]
