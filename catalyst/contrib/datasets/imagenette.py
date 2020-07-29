# flake8: noqa
# TODO: docs and refactor for datasets-contrib
# This subpackage was borrowed from
# [torchvision](https://github.com/pytorch/vision).
import codecs
import os

import numpy as np

import torch
from torch.utils.data import Dataset

from catalyst
from catalyst.contrib.datasets.utils import download_and_extract_archive


import glob
from catalyst.data import PathsDataset
from catalyst.contrib.data.cv import ImageReader
from catalyst import utils
import functools
class ImageFolderDataset(PathsDataset):
    def __init__(
        self, rootpath: str, transform: Optional[Callable[[Dict], Dict]] = None
    ) -> None:
        files = glob.iglob(f"{rootpath}/**/*", recursive=True)
        images = filter(utils.has_image_extension, files)

        super().__init__(
            filenames=sorted(images),
            open_fn=ImageReader(input_key="image", rootpath=rootpath),
            label_fn=(lambda fn: Path(fn.parent.name)),
            dict_transform=transform,
        )


class Imagenette():
    resources = {
        "160px": {
            "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "md5": None,
        },
        "320px": {
            "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "md5": None,
        },
    }

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ) -> None:
        # download dataset if needed
        if download and not self._check_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            os.makedirs(self.processed_folder, exist_ok=True)

            # download files
            for url, md5 in self.resources:
                filename = url.rpartition("/")[2]
                download_and_extract_archive(
                    url,
                    download_root=self.raw_folder,
                    filename=filename,
                    md5=md5,
                )

    # def __repr__(self):
    #     head = "Dataset " + self.__class__.__name__
    #     body = ["Number of datapoints: {}".format(self.__len__())]
    #     if self.root is not None:
    #         body.append("Root location: {}".format(self.root))
    #     body += self.extra_repr().splitlines()
    #     if hasattr(self, "transforms") and self.transforms is not None:
    #         body += [repr(self.transforms)]
    #     lines = [head] + [" " * self._repr_indent + line for line in body]
    #     return "\n".join(lines)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def download(self):
        # process and save as torch files
        print("Processing...")

        training_set = (
            read_image_file(
                os.path.join(self.raw_folder, "train-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.raw_folder, "train-labels-idx1-ubyte")
            ),
        )
        test_set = (
            read_image_file(
                os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")
            ),
        )
        with open(
            os.path.join(self.processed_folder, self.training_file), "wb"
        ) as f:
            torch.save(training_set, f)
        with open(
            os.path.join(self.processed_folder, self.test_file), "wb"
        ) as f:
            torch.save(test_set, f)

        print("Done!")
