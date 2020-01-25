import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform
import torch


class TensorToImage(ImageOnlyTransform):
    """
    Casts torch.tensor to numpy array
    """
    def __init__(self, always_apply=False, p=1.0):
        """
        Args:
            always_apply (bool): need to apply this transform anyway
            p (float): probability for this transform
        """
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        """Apply the transform to the image"""
        return np.expand_dims(img.numpy(), axis=0)


class ToTensor(ImageOnlyTransform):
    """
    Casts numpy array to torch.tensor, but do not move channels dim
    """
    def __init__(self, always_apply=False, p=1.0):
        """
        Args:
            always_apply (bool): need to apply this transform anyway
            p (float): probability for this transform
        """
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        """Apply the transform to the image"""
        return torch.from_numpy(img)


__all__ = [
    "TensorToImage",
    "ToTensor"
]
