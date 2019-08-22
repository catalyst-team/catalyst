# !/usr/bin/env python
# title           :process_images
# description     :script to resize images and clear exif in parallel
# author          :Sergey Kolesnikov, Vsevolod Poletaev, Roman Tezikov
# author_email    :scitator@gmail.com, poletaev.va@gmail.com, tez.romach@gmail.com
# date            :20190822
# version         :19.08.7
# * equal contribution
# ==============================================================================


import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from albumentations.augmentations.functional import longest_max_size

from catalyst.utils import boolean_flag, imread, imwrite, \
    Pool, tqdm_parallel_imap, get_pool

# Limit cv2's processor usage
# cv2.setNumThreads() doesn't work
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)


def build_args(parser):
    parser.add_argument("--in-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-workers", "-j", type=int, default=4)

    parser.add_argument("--max-side", default=None, type=int)
    boolean_flag(parser, "clear-exif", default=True)
    boolean_flag(parser, "grayscale", default=False)
    boolean_flag(parser, "expand-dims", default=True)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


class Preprocessor:
    def __init__(
        self,
        in_dir: Path,
        out_dir: Path,
        max_side: int = None,
        clear_exif: bool = True,
        grayscale: bool = False,
        expand_dims: bool = True,
        interpolation=cv2.INTER_LANCZOS4
    ):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.grayscale = grayscale
        self.expand_dims = expand_dims
        self.max_side = max_side
        self.clear_exif = clear_exif
        self.interpolation = interpolation

    def preprocess(self, image_path: Path):
        target_path = self.out_dir / image_path.relative_to(self.in_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.array(
            imread(
                uri=image_path,
                grayscale=self.grayscale,
                expand_dims=self.expand_dims,
                exifrotate=self.clear_exif))

        if self.max_side is not None:
            image = longest_max_size(image, self.max_side, self.interpolation)

        imwrite(target_path, image)

    def process_all(self, pool: Pool):
        images = [*self.in_dir.glob("**/*.jpg")]
        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    args = args.__dict__
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
