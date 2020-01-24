import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform
import torch


class TensorToImage(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return np.expand_dims(img.numpy(), axis=0)


class ToTensor(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return torch.from_numpy(img)

__all__ = [
    "TensorToImage",
    "ToTensor"
]
