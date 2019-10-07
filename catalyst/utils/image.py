from typing import List, Tuple
import logging
import os
import tempfile
import numpy as np
import imageio
from skimage.color import label2rgb, rgb2gray

import torch

_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)

logger = logging.getLogger(__name__)

JPEG4PY_ENABLED = False
if os.environ.get("FORCE_JPEG_TURBO", False):
    try:
        import jpeg4py as jpeg

        # check libjpeg-turbo availability through image reading
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as fp:
            imageio.imwrite(fp.name, img)
            img = jpeg.JPEG(fp.name).decode()

        JPEG4PY_ENABLED = True
    except ImportError:
        logger.warning(
            "jpeg4py not available. "
            "To install jpeg4py, run `pip install jpeg4py`."
        )
    except OSError:
        logger.warning(
            "libjpeg-turbo not available. "
            "To install libjpeg-turbo, run `apt-get install libturbojpeg`."
        )


def imread(uri, grayscale=False, expand_dims=True, rootpath=None, **kwargs):
    """

    Args:
        uri: {str, pathlib.Path, bytes, file}
        The resource to load the image from, e.g. a filename, pathlib.Path,
        http address or file object, see the docs for more info.
        grayscale:
        expand_dims:
        rootpath:

    Returns:

    """
    if rootpath is not None:
        uri = (
            uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)
        )

    if JPEG4PY_ENABLED and uri.endswith(("jpg", "JPG", "jpeg", "JPEG")):
        img = jpeg.JPEG(uri).decode()
        if grayscale:
            img = rgb2gray(img)
    else:
        img = imageio.imread(uri, as_gray=grayscale, pilmode="RGB", **kwargs)

    if expand_dims and len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    return img


imwrite = imageio.imwrite
imsave = imageio.imsave


def mimwrite_with_meta(uri, ims, meta, **kwargs):
    writer = imageio.get_writer(uri, mode="I", **kwargs)
    writer.set_meta_data(meta)
    with writer:
        for i in ims:
            writer.append_data(i)


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


def mask_to_overlay_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    threshold: float = 0,
    mask_strength: float = 0.5
) -> np.ndarray:
    """
    Draws every mask for with some color over image

    Args:
        image (np.ndarray): RGB image used as underlay for masks
        masks (List[np.ndarray]): list of masks
        threshold (float): threshold for masks binarization
        mask_strength (float): opacity of colorized masks
    Returns:
        np.ndarray: HxWx3 image with overlay

    """
    h, w = image.shape[:2]
    labels = np.zeros((h, w), np.uint8)

    for idx, mask in enumerate(masks, start=1):
        labels[mask > threshold] = idx

    mask = label2rgb(labels, bg_label=0)

    image = np.array(image) / 255.0
    image_with_overlay = image * (1 - mask_strength) + mask * mask_strength
    image_with_overlay = (
        (image_with_overlay * 255).clip(0, 255).round().astype(np.uint8)
    )

    return image_with_overlay
