from typing import List, Tuple, Union
import logging
import os
import pathlib
import tempfile

import imageio
import numpy as np
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


def imread(
    uri,
    grayscale: bool = False,
    expand_dims: bool = True,
    rootpath: Union[str, pathlib.Path] = None,
    **kwargs,
) -> np.ndarray:
    """Reads an image from the specified file.

    Args:
        uri (str, pathlib.Path, bytes, file): the resource to load the image
        from, e.g. a filename, ``pathlib.Path``, http address or file object,
        see ``imageio.imread`` docs for more info
        grayscale (bool):
        expand_dims (bool):
        rootpath (Union[str, pathlib.Path]): path to the resource with image
            (allows to use relative path)

    Returns:
        np.ndarray: image
    """
    uri = str(uri)

    if rootpath is not None:
        rootpath = str(rootpath)
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    if JPEG4PY_ENABLED and uri.endswith(("jpg", "JPG", "jpeg", "JPEG")):
        img = jpeg.JPEG(uri).decode()
    else:
        # @TODO: add tiff support, currently – jpg and png
        img = imageio.imread(uri, as_gray=grayscale, pilmode="RGB", **kwargs)
    if grayscale:
        img = rgb2gray(img)

    if expand_dims and len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    return img


imwrite = imageio.imwrite
imsave = imageio.imsave


def mimread(
    uri,
    clip_range: Tuple[int, int] = None,
    expand_dims: bool = True,
    rootpath: Union[str, pathlib.Path] = None,
    **kwargs,
) -> np.ndarray:
    """Reads multiple images from the specified file.

    Args:
        uri (str, pathlib.Path, bytes, file): the resource to load the image
        from, e.g. a filename, ``pathlib.Path``, http address or file object,
        see ``imageio.mimread`` docs for more info
        clip_range (Tuple[int, int]): lower and upper interval edges,
            image values outside the interval are clipped to the interval edges
        expand_dims (bool): if True, append channel axis to grayscale images
        rootpath (Union[str, pathlib.Path]): path to the resource with image
            (allows to use relative path)

    Returns:
        np.ndarray: image
    """
    if rootpath is not None:
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    image = np.dstack(imageio.mimread(uri, **kwargs))
    if clip_range is not None:
        image = np.clip(image, *clip_range)

    if expand_dims and len(image.shape) < 3:  # grayscale
        image = np.expand_dims(image, -1)

    return image


def mimwrite_with_meta(uri, ims, meta, **kwargs):
    """@TODO: Docs. Contribution is welcome."""
    writer = imageio.get_writer(uri, mode="I", **kwargs)
    writer.set_meta_data(meta)
    with writer:
        for i in ims:
            writer.append_data(i)


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


def mask_to_overlay_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    threshold: float = 0,
    mask_strength: float = 0.5,
) -> np.ndarray:
    """Draws every mask for with some color over image.

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


def has_image_extension(uri) -> bool:
    """Check that file has image extension.

    Args:
        uri (Union[str, pathlib.Path]): the resource to load the file from

    Returns:
        bool: True if file has image extension, False otherwise
    """
    _, ext = os.path.splitext(uri)
    return ext.lower() in {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}


__all__ = [
    "has_image_extension",
    "imread",
    "imwrite",
    "imsave",
    "mask_to_overlay_image",
    "mimread",
    "mimwrite_with_meta",
    "tensor_from_rgb_image",
    "tensor_to_ndimage",
]
