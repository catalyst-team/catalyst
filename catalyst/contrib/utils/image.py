# flake8: noqa
from typing import Tuple, Union
import logging
import os
import pathlib
import tempfile

import imageio
import numpy as np
from skimage.color import rgb2gray  # label2rgb

from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

if SETTINGS.use_libjpeg_turbo:
    try:
        import jpeg4py as jpeg

        # check libjpeg-turbo availability through image reading
        _test_img = np.zeros((1, 1, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as fp:
            imageio.imwrite(fp.name, _test_img)
            _test_img = jpeg.JPEG(fp.name).decode()

    except ImportError as ex:
        logger.warning("jpeg4py not available. " "To install jpeg4py, run `pip install jpeg4py`.")
        raise ex
    except OSError as ex:
        logger.warning(
            "libjpeg-turbo not available. "
            "To install libjpeg-turbo, run `apt-get install libturbojpeg`."
        )
        raise ex


def imread(
    uri,
    grayscale: bool = False,
    expand_dims: bool = True,
    rootpath: Union[str, pathlib.Path] = None,
    **kwargs,
) -> np.ndarray:
    """
    Reads an image from the specified file.

    Args:
        uri (str, pathlib.Path, bytes, file): the resource to load the image
          from, e.g. a filename, ``pathlib.Path``, http address or file object,
          see ``imageio.imread`` docs for more info
        grayscale: if True, make all images grayscale
        expand_dims: if True, append channel axis to grayscale images
          rootpath (Union[str, pathlib.Path]): path to the resource with image
          (allows to use relative path)
        rootpath (Union[str, pathlib.Path]): path to the resource with image
            (allows to use relative path)
        **kwargs: extra params for image read

    Returns:
        np.ndarray: image
    """
    uri = str(uri)

    if rootpath is not None:
        rootpath = str(rootpath)
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    if SETTINGS.use_libjpeg_turbo and uri.endswith(("jpg", "JPG", "jpeg", "JPEG")):
        img = jpeg.JPEG(uri).decode()
    else:
        # @TODO: add tiff support, currently – jpg and png
        img = imageio.imread(uri, as_gray=grayscale, pilmode="RGB", **kwargs)

    if grayscale:
        img = rgb2gray(img)

    if expand_dims and len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    return img


def imwrite(**kwargs):
    """
    ``imwrite(uri, im, format=None, **kwargs)``

    Write an image to the specified file.
    Alias for ``imageio.imwrite``.

    Args:
        **kwargs: parameters for ``imageio.imwrite``

    Returns:
        image save result
    """
    return imageio.imwrite(**kwargs)


def imsave(**kwargs):
    """
    ``imwrite(uri, im, format=None, **kwargs)``

    Write an image to the specified file.
    Alias for ``imageio.imsave``.

    Args:
        **kwargs: parameters for ``imageio.imsave``

    Returns:
        image save result
    """
    return imageio.imsave(**kwargs)


def mimread(
    uri,
    clip_range: Tuple[int, int] = None,
    expand_dims: bool = True,
    rootpath: Union[str, pathlib.Path] = None,
    **kwargs,
) -> np.ndarray:
    """
    Reads multiple images from the specified file.

    Args:
        uri (str, pathlib.Path, bytes, file): the resource to load the image
          from, e.g. a filename, ``pathlib.Path``, http address or file object,
          see ``imageio.mimread`` docs for more info
        clip_range (Tuple[int, int]): lower and upper interval edges,
          image values outside the interval are clipped to the interval edges
        expand_dims: if True, append channel axis to grayscale images
          rootpath (Union[str, pathlib.Path]): path to the resource with image
          (allows to use relative path)
        rootpath (Union[str, pathlib.Path]): path to the resource with image
            (allows to use relative path)
        **kwargs: extra params for image read

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


# def mimwrite_with_meta(uri, ims, meta, **kwargs):
#     """@TODO: Docs. Contribution is welcome."""
#     writer = imageio.get_writer(uri, mode="I", **kwargs)
#     writer.set_meta_data(meta)
#     with writer:
#         for i in ims:
#             writer.append_data(i)


# def mask_to_overlay_image(
#     image: np.ndarray, masks: List[np.ndarray], threshold: float = 0, mask_strength: float = 0.5,
# ) -> np.ndarray:
#     """Draws every mask for with some color over image.
#
#     Args:
#         image: RGB image used as underlay for masks
#         masks: list of masks
#         threshold: threshold for masks binarization
#         mask_strength: opacity of colorized masks
#
#     Returns:
#         np.ndarray: HxWx3 image with overlay
#     """
#     h, w = image.shape[:2]
#     labels = np.zeros((h, w), np.uint8)
#
#     for idx, mask in enumerate(masks, start=1):
#         labels[mask > threshold] = idx
#
#     mask = label2rgb(labels, bg_label=0)
#
#     image = np.array(image) / 255.0
#     image_with_overlay = image * (1 - mask_strength) + mask * mask_strength
#     image_with_overlay = (image_with_overlay * 255).clip(0, 255).round().astype(np.uint8)
#
#     return image_with_overlay


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
    # "mask_to_overlay_image",
    "mimread",
    # "mimwrite_with_meta",
]
