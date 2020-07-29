from typing import Iterable, Tuple
import os

from catalyst.contrib.data.cv.dataset import ImageFolderDataset
from catalyst.contrib.datasets.utils import download_and_extract_archive


class ImageClassificationDataset(ImageFolderDataset):
    name: str
    resources: Iterable[Tuple[str, str]] = None

    def __init__(
        self, root: str, train: bool = True, download: bool = False, **kwargs
    ):
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
