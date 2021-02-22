#!/usr/bin/env python
# usage:
# catalyst-contrib process-images \
#   --in-dir ./data_in \
#   --out-dir ./data_out \
#   --num-workers 4 \
#   --max-size 224 \
#   --clear-exif \
#   --grayscale

from typing import List
import argparse
from functools import wraps
from multiprocessing.pool import Pool
import os
from pathlib import Path

import cv2
import numpy as np

from catalyst.contrib.utils.image import has_image_extension, imread, imwrite
from catalyst.contrib.utils.parallel import get_pool, tqdm_parallel_imap
from catalyst.utils.misc import boolean_flag


def build_args(parser):
    """
    Constructs the command-line arguments for
    ``catalyst-contrib process-images``.

    Args:
        parser: current parser

    Returns:
        updated parser
    """
    parser.add_argument("--in-dir", required=True, type=Path, help="Raw data folder path")

    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Processed images folder path",
    )

    parser.add_argument(
        "--num-workers",
        "-j",
        default=1,
        type=int,
        help="Number of workers to parallel the processing",
    )

    parser.add_argument(
        "--max-size",
        default=None,
        required=False,
        type=int,
        help="Output images size. E.g. 224, 448",
    )

    boolean_flag(parser, "clear-exif", default=True, help="Clear EXIF data")

    boolean_flag(parser, "grayscale", default=False, help="Read images in grayscale")

    boolean_flag(
        parser, "expand-dims", default=True, help="Expand array shape for grayscale images",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


# <--- taken from albumentations - https://github.com/albu/albumentations --->


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))  # noqa: WPS432

    return int(round(number))


def preserve_channel_dim(func):
    """Preserve dummy channel dim."""  # noqa: D202

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def _func_max_size(img, max_size, interpolation, func):
    height, width = img.shape[:2]

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        out_size = tuple(py3round(dim * scale) for dim in (width, height))
        img = cv2.resize(img, out_size, interpolation=interpolation)
    return img


@preserve_channel_dim
def longest_max_size(img, max_size, interpolation):
    """@TODO: Docs. Contribution is welcome."""
    return _func_max_size(img, max_size, interpolation, max)


# <--- taken from albumentations - https://github.com/albu/albumentations --->


class Preprocessor:
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_dir: Path,
        out_dir: Path,
        max_size: int = None,
        clear_exif: bool = True,
        grayscale: bool = False,
        expand_dims: bool = True,
        interpolation=cv2.INTER_LANCZOS4,
    ):
        """@TODO: Docs. Contribution is welcome."""
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.grayscale = grayscale
        self.expand_dims = expand_dims
        self.max_size = max_size
        self.clear_exif = clear_exif
        self.interpolation = interpolation

    def preprocess(self, image_path: Path):
        """@TODO: Docs. Contribution is welcome."""
        try:
            _, extension = os.path.splitext(image_path)
            kwargs = {
                "grayscale": self.grayscale,
                "expand_dims": self.expand_dims,
            }
            if extension.lower() in {"jpg", "jpeg"}:
                # imread does not have exifrotate for non-jpeg type
                kwargs["exifrotate"] = not self.clear_exif

            image = np.array(imread(uri=image_path, **kwargs))
        except Exception as e:
            print(f"Cannot read file {image_path}, exception: {e}")
            return

        if self.max_size is not None:
            image = longest_max_size(image, self.max_size, self.interpolation)

        target_path = self.out_dir / image_path.relative_to(self.in_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = image.clip(0, 255).round().astype(np.uint8)  # noqa: WPS432
        imwrite(target_path, image)

    def process_all(self, pool: Pool):
        """@TODO: Docs. Contribution is welcome."""
        images: List[Path] = []
        for root, _, files in os.walk(self.in_dir):
            root = Path(root)
            images.extend([root / filename for filename in files if has_image_extension(filename)])

        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    """Run the ``catalyst-contrib process-images`` script."""
    args = args.__dict__
    args.pop("command", None)
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    # Limit cv2's processor usage
    # cv2.setNumThreads() doesn't work
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.ocl.setUseOpenCL(False)

    args = parse_args()
    main(args)
