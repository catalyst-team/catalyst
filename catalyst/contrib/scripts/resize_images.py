# !/usr/bin/env python
# title           :resize_images
# description     :script to resize images in parallel
# author          :Vsevolod Poletaev, Sergey Kolesnikov
# author_email    :poletaev.va@gmail.com, scitator@gmail.com
# date            :20190822
# version         :19.08.7
# ==============================================================================


import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio
import numpy as np
from albumentations.augmentations.functional import longest_max_size

from catalyst.utils.parallel import Pool, tqdm_parallel_imap, get_pool

# Limit cv2's processor usage
# cv2.setNumThreads() doesn't work
os.environ["OMP_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)


class Preprocessor:
    def __init__(
        self,
        source: Path,
        target: Path,
        max_side: int,
        interpolation=cv2.INTER_LANCZOS4
    ):
        self.target = target
        self.source = source
        self.interpolation = interpolation
        self.max_side = max_side

    def preprocess(self, image_path: Path):
        target_path = self.target / image_path.relative_to(self.source)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.array(imageio.imread(image_path))
        image = longest_max_size(image, self.max_side, self.interpolation)

        imageio.imwrite(target_path, image)

    def process_all(self, pool: Pool):
        images = [*self.source.glob("**/*.jpg")]
        tqdm_parallel_imap(self.preprocess, images, pool)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)

    parser.add_argument("--max-side", default=1024, type=int)
    parser.add_argument("--num-workers", "-j", type=int, default=4)

    args = parser.parse_args().__dict__

    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)
