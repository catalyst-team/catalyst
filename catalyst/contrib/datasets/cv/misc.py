from typing import Iterable, Tuple
import os

from catalyst.contrib.data.cv.dataset import ImageFolderDataset
from catalyst.contrib.datasets.functional import download_and_extract_archive


class ImageClassificationDataset(ImageFolderDataset):
    """
    Base class for datasets with the following structure:

    .. code-block:: bash

        path/to/dataset/
        |-- train/
        |   |-- class1/  # folder of N images
        |   |   |-- train_image11
        |   |   |-- train_image12
        |   |   ...
        |   |   `-- train_image1N
        |   ...
        |   `-- classM/  # folder of K images
        |       |-- train_imageM1
        |       |-- train_imageM2
        |       ...
        |       `-- train_imageMK
        `-- val/
            |-- class1/  # folder of P images
            |   |-- val_image11
            |   |-- val_image12
            |   ...
            |   `-- val_image1P
            ...
            `-- classM/  # folder of T images
                |-- val_imageT1
                |-- val_imageT2
                ...
                `-- val_imageMT

    """

    # name of dataset folder
    name: str

    # list of (url, md5 hash) tuples representing files to download
    resources: Iterable[Tuple[str, str]] = None

    def __init__(self, root: str, train: bool = True, download: bool = False, **kwargs):
        """Constructor method for the ``ImageClassificationDataset`` class.

        Args:
            root: root directory of dataset
            train: if ``True``, creates dataset from ``train/``
                subfolder, otherwise from ``val/``
            download: if ``True``, downloads the dataset from
                the internet and puts it in root directory. If dataset
                is already downloaded, it is not downloaded again
            **kwargs:
        """
        # downlad dataset if needed
        if download and not os.path.exists(os.path.join(root, self.name)):
            os.makedirs(root, exist_ok=True)

            # download files
            for url, md5 in self.resources:
                filename = url.rpartition("/")[2]
                download_and_extract_archive(url, download_root=root, filename=filename, md5=md5)

        rootpath = os.path.join(root, self.name, "train" if train else "val")
        super().__init__(rootpath=rootpath, **kwargs)


__all__ = ["ImageClassificationDataset"]
