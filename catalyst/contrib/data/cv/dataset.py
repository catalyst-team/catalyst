from typing import Callable, Dict, Mapping, Optional
from typing import Iterable, Tuple
import os
import glob
from pathlib import Path

from catalyst.contrib.data.cv.reader import ImageReader
from catalyst.contrib.data.reader import ReaderCompose, ScalarReader
from catalyst.contrib.utils.cv.image import has_image_extension
from catalyst.data.dataset.torch import PathsDataset
from catalyst.contrib.datasets.functional import download_and_extract_archive


class ImageFolderDataset(PathsDataset):
    """
    Dataset class that derives targets from samples filesystem paths.
    Dataset structure should be the following:

    .. code-block:: bash

        rootpat/
        |-- class1/  # folder of N images
        |   |-- image11
        |   |-- image12
        |   ...
        |   `-- image1N
        ...
        `-- classM/  # folder of K images
            |-- imageM1
            |-- imageM2
            ...
            `-- imageMK

    """

    def __init__(
        self,
        rootpath: str,
        target_key: str = "targets",
        dir2class: Optional[Mapping[str, int]] = None,
        dict_transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        """Constructor method for the :class:`ImageFolderDataset` class.

        Args:
            rootpath: root directory of dataset
            target_key: key to use to store target label
            dir2class (Mapping[str, int], optional): mapping from folder name
                to class index
            dict_transform (Callable[[Dict], Dict]], optional): transforms
                to use on dict
        """
        files = glob.iglob(f"{rootpath}/**/*")
        images = sorted(filter(has_image_extension, files))

        if dir2class is None:
            dirs = sorted({Path(f).parent.name for f in images})
            dir2class = {dirname: index for index, dirname in enumerate(dirs)}

        super().__init__(
            filenames=images,
            open_fn=ReaderCompose(
                [
                    ImageReader(input_key="image", rootpath=rootpath),
                    ScalarReader(
                        input_key=target_key,
                        output_key=target_key,
                        dtype=int,
                        default_value=-1,
                    ),
                ]
            ),
            label_fn=lambda fn: dir2class[Path(fn).parent.name],
            features_key="image",
            target_key=target_key,
            dict_transform=dict_transform,
        )

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

    def __init__(
        self, root: str, train: bool = True, download: bool = False, **kwargs
    ):
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
                download_and_extract_archive(
                    url, download_root=root, filename=filename, md5=md5
                )

        rootpath = os.path.join(root, self.name, "train" if train else "val")
        super().__init__(rootpath=rootpath, **kwargs)



__all__ = ["ImageFolderDataset", "ImageClassificationDataset"]
