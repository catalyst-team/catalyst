import numpy as np

from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
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


class ToTensor(ToTensorV2):
    """
    Casts numpy array to ``torch.tensor``
    """
    def __init__(
        self,
        move_channels_dim: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Args:
            move_channels_dim (bool): if ``False``, casts numpy array
                to ``torch.tensor``, but do not move channels dim
            always_apply (bool): need to apply this transform anyway
            p (float): probability for this transform
        """
        super().__init__(always_apply, p)
        self.move_channels_dim = move_channels_dim

    def apply(self, img, **params):
        """Apply the transform to the image"""
        if self.move_channels_dim:
            return super().apply(img, **params)
        else:
            return torch.from_numpy(img)

    def apply_to_mask(self, mask: np.ndarray, **params) -> torch.Tensor:
        """Apply the transform to the mask"""
        if self.move_channels_dim:
            mask = mask.transpose(2, 0, 1)
        return super().apply_to_mask(mask.astype(np.float32), **params)

    def get_transform_init_args_names(self):
        return ("move_channels_dim",)


__all__ = [
    "TensorToImage",
    "ToTensor"
]
