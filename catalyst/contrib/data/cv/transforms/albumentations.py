from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

from catalyst.contrib.utils.cv.tensor import tensor_to_ndimage


class TensorToImage(ImageOnlyTransform):
    """Casts ``torch.tensor`` to ``numpy.array``."""

    def __init__(
        self,
        denormalize: bool = False,
        move_channels_dim: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Args:
            denormalize: if True, multiply image(s) by ImageNet std and
                add ImageNet mean
            move_channels_dim: if True, convert [B]xCxHxW tensor
                to [B]xHxWxC format
            always_apply: need to apply this transform anyway
            p: probability for this transform
        """
        super().__init__(always_apply, p)
        self.denormalize = denormalize
        self.move_channels_dim = move_channels_dim

    def apply(self, img: torch.Tensor, **params) -> np.ndarray:
        """Apply the transform to the image."""
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        return tensor_to_ndimage(
            img, denormalize=self.denormalize, move_channels_dim=self.move_channels_dim,
        )


class ImageToTensor(ToTensorV2):
    """Casts ``numpy.array`` to ``torch.tensor``."""

    def __init__(
        self, move_channels_dim: bool = True, always_apply: bool = False, p: float = 1.0,
    ):
        """
        Args:
            move_channels_dim: if ``False``, casts numpy array
                to ``torch.tensor``, but do not move channels dim
            always_apply: need to apply this transform anyway
            p: probability for this transform
        """
        super().__init__(always_apply, p)
        self.move_channels_dim = move_channels_dim

    def apply(self, img: np.ndarray, **params) -> torch.Tensor:
        """Apply the transform to the image."""
        if self.move_channels_dim:
            return super().apply(img, **params)
        return torch.from_numpy(img)

    def apply_to_mask(self, mask: np.ndarray, **params) -> torch.Tensor:
        """Apply the transform to the mask."""
        if self.move_channels_dim:
            mask = mask.transpose(2, 0, 1)
        return super().apply_to_mask(mask.astype(np.float32), **params)

    def get_transform_init_args_names(self) -> tuple:
        """Get transform init args names."""
        return ("move_channels_dim",)


__all__ = ["TensorToImage", "ImageToTensor"]
