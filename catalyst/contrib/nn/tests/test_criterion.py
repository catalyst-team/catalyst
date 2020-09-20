# flake8: noqa
import numpy as np

import torch

from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import (
    CircleLoss,
    TripletMarginLossWithSampler,
)
from catalyst.data import AllTripletsSampler


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for module_class in module.__dict__.values():
        if isinstance(module_class, type):
            if module_class == CircleLoss:
                instance = module_class(margin=0.25, gamma=256)
            elif module_class == TripletMarginLossWithSampler:
                instance = module_class(
                    margin=1.0, sampler_inbatch=AllTripletsSampler()
                )
            else:
                instance = module_class()
            assert instance is not None
