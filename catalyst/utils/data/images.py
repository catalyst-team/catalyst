from typing import List, Tuple
import os
import numpy as np
import cv2
import jpeg4py as jpeg
from skimage.color import label2rgb

import torch

_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def read_image(image_name, datapath=None, grayscale=False):
    if datapath is not None:
        image_name = (
            image_name if image_name.startswith(datapath) else
            os.path.join(datapath, image_name)
        )

    img = None
    try:
        if image_name.endswith(("jpg", "JPG", "jpeg", "JPEG")):
            img = jpeg.JPEG(image_name).decode()
    except Exception:
        pass

    if img is None:
        img = cv2.imread(image_name)

        if len(img.shape) == 3:  # BGR -> RGB
            img = img[:, :, ::-1]

    if len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    if img.shape[-1] != 3 and not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


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


def binary_mask_to_overlay_image(image: np.ndarray, masks: List[np.ndarray]):
    """Draws every mask for with some color over image"""
    h, w = image.shape[:2]
    labels = np.zeros((h, w), np.uint8)

    for idx, mask in enumerate(masks):
        labels[mask > 0] = idx + 1

    image_with_overlay = label2rgb(labels, image)

    image_with_overlay = (image_with_overlay * 255).round().astype(np.uint8)
    return image_with_overlay
