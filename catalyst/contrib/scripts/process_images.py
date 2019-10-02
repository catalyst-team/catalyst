#!/usr/bin/env python
# usage:
# catalyst-data process-images \
#   --in-dir ./data_in \
#   --out-dir ./data_out \
#   --num-workers 4 \
#   --max-size 224 \
#   --clear-exif \
#   --grayscale

import os
import argparse
from pathlib import Path
from functools import wraps

import cv2
import numpy as np

from catalyst.utils import boolean_flag, imread, imwrite, \
    Pool, tqdm_parallel_imap, get_pool

# Limit cv2's processor usage
# cv2.setNumThreads() doesn't work
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)


def build_args(parser):
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Raw data folder path")

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Processed images folder path")

    parser.add_argument(
        "--num-workers", "-j",
        default=1,
        type=int,
        help="Number of workers to parallel the processing")

    parser.add_argument(
        "--extension",
        default="jpg",
        type=str,
        help="Input images extension. JPG is default.")

    parser.add_argument(
        "--max-size",
        default=None,
        required=True,
        type=int,
        help="Output images size. E.g. 224, 448")

    boolean_flag(
        parser,
        "clear-exif",
        default=True,
        help="Clear EXIF data")

    boolean_flag(
        parser,
        "grayscale",
        default=False,
        help="Read images in grayscale")

    boolean_flag(
        parser,
        "expand-dims",
        default=True,
        help="Expand array shape for grayscale images")

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


# <--- taken from albumentations - https://github.com/albu/albumentations --->

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def preserve_channel_dim(func):
    """Preserve dummy channel dim."""
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
    return _func_max_size(img, max_size, interpolation, max)

# <--- taken from albumentations - https://github.com/albu/albumentations --->


class Preprocessor:
    def __init__(
        self,
        in_dir: Path,
        out_dir: Path,
        max_size: int = None,
        clear_exif: bool = True,
        grayscale: bool = False,
        expand_dims: bool = True,
        extension: str = "jpg",
        interpolation=cv2.INTER_LANCZOS4,
    ):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.grayscale = grayscale
        self.expand_dims = expand_dims
        self.max_size = max_size
        self.clear_exif = clear_exif
        self.extension = extension
        self.interpolation = interpolation

    def preprocess(self, image_path: Path):
        try:
            if self.extension in ("jpg", "JPG", "jpeg", "JPEG"):
                image = np.array(
                    imread(
                        uri=image_path,
                        grayscale=self.grayscale,
                        expand_dims=self.expand_dims,
                        exifrotate=not self.clear_exif))
            else:  # imread does not have exifrotate for non-jpeg type
                image = np.array(
                    imread(
                        uri=image_path,
                        grayscale=self.grayscale,
                        expand_dims=self.expand_dims))
        except Exception as e:
            print(f"Cannot read file {image_path}, exception: {e}")
            return

        if self.max_size is not None:
            image = longest_max_size(image, self.max_size, self.interpolation)

        target_path = self.out_dir / image_path.relative_to(self.in_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = image.clip(0, 255).round().astype(np.uint8)
        imwrite(target_path, image)

    def process_all(self, pool: Pool):
        images = [*self.in_dir.glob(f"**/*.{self.extension}")]
        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    args = args.__dict__
    args.pop("command", None)
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
