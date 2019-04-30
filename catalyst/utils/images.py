from typing import Tuple

import torch
import numpy as np

_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def tensor_to_ndimage(
    images: torch.Tensor,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
    dtype=np.float32
) -> np.ndarray:
    """
    Convert float image(s) with standard normalization to
    np.ndarray with [0..1] when dtype is np.float32 and [0..255]
    when dtype is `np.uint8`.

    Args:
        images: [B]xCxHxW float tensor
        mean: mean to add
        std: std to multiply
        dtype: result ndarray dtype. Only float32 and uint8 are supported.
    Returns:
        [B]xHxWxC np.ndarray of dtype
    """
    has_batch_dim = len(images.shape) == 4

    num_shape = (3, 1, 1)

    if has_batch_dim:
        num_shape = (1, ) + num_shape

    mean = images.new_tensor(mean).view(*num_shape)
    std = images.new_tensor(std).view(*num_shape)

    images = images * std + mean

    images = images.clamp(0, 1).numpy()

    images = np.moveaxis(images, -3, -1)

    if dtype == np.uint8:
        images = (images * 255).round().astype(dtype)
    else:
        assert dtype == np.float32, "Only float32 and uint8 are supported"

    return images


__all__ = ["tensor_to_ndimage"]
