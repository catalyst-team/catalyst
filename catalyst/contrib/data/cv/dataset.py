from typing import Callable, Dict, Mapping, Optional
import glob
from pathlib import Path

from catalyst.contrib.data.cv.reader import ImageReader
from catalyst.contrib.data.reader import ReaderCompose, ScalarReader
from catalyst.contrib.utils.image import has_image_extension
from catalyst.data.dataset.torch import PathsDataset


class ImageFolderDataset(PathsDataset):
    """
    Dataset class that derives targets from samples filesystem paths.
    Dataset structure should be the following:

    .. code-block:: bash

        rootpath/
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
                        input_key=target_key, output_key=target_key, dtype=int, default_value=-1,
                    ),
                ]
            ),
            label_fn=lambda fn: dir2class[Path(fn).parent.name],
            features_key="image",
            target_key=target_key,
            dict_transform=dict_transform,
        )


__all__ = ["ImageFolderDataset"]
