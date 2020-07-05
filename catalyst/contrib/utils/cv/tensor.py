# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Tuple

import numpy as np

import torch

_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    """@TODO: Docs. Contribution is welcome."""
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def tensor_to_ndimage(
    images: torch.Tensor,
    denormalize: bool = True,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
    move_channels_dim: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert float image(s) with standard normalization to
    np.ndarray with [0..1] when dtype is np.float32 and [0..255]
    when dtype is `np.uint8`.

    Args:
        images (torch.Tensor): [B]xCxHxW float tensor
        denormalize (bool): if True, multiply image(s) by std and add mean
        mean (Tuple[float, float, float]): per channel mean to add
        std (Tuple[float, float, float]): per channel std to multiply
        move_channels_dim (bool): if True, convert tensor to [B]xHxWxC format
        dtype: result ndarray dtype. Only float32 and uint8 are supported

    Returns:
        [B]xHxWxC np.ndarray of dtype
    """
    if denormalize:
        has_batch_dim = len(images.shape) == 4

        mean = images.new_tensor(mean).view(
            *((1,) if has_batch_dim else ()), len(mean), 1, 1
        )
        std = images.new_tensor(std).view(
            *((1,) if has_batch_dim else ()), len(std), 1, 1
        )

        images = images * std + mean

    images = images.clamp(0, 1).numpy()

    if move_channels_dim:
        images = np.moveaxis(images, -3, -1)

    if dtype == np.uint8:
        images = (images * 255).round().astype(dtype)
    else:
        assert dtype == np.float32, "Only float32 and uint8 are supported"

    return images


__all__ = ["tensor_from_rgb_image", "tensor_to_ndimage"]
