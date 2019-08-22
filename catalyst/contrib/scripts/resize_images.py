# !/usr/bin/env python
# title           :resize_images
# description     :script to resize images in parallel
# author          :Vsevolod Poletaev, Sergey Kolesnikov
# author_email    :poletaev.va@gmail.com, scitator@gmail.com
# date            :20190822
# version         :19.08.7
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
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)

    parser.add_argument("--max-side", default=224, type=int)
    parser.add_argument("--num-workers", "-j", type=int, default=4)

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
        source: Path,
        target: Path,
        max_side: int,
        grayscale: bool = False,
        expand_dims: bool = True,
        interpolation=cv2.INTER_LANCZOS4
    ):
        self.target = target
        self.source = source
        self.grayscale = grayscale
        self.expand_dims = expand_dims
        self.max_side = max_side
        self.interpolation = interpolation

    def preprocess(self, image_path: Path):
        target_path = self.target / image_path.relative_to(self.source)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.array(
            imread(
                uri=image_path,
                grayscale=self.grayscale,
                expand_dims=self.expand_dims))
        image = longest_max_size(image, self.max_side, self.interpolation)

        imwrite(target_path, image)

    def process_all(self, pool: Pool):
        images = [*self.source.glob("**/*.jpg")]
        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    args = args.__dict__
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
