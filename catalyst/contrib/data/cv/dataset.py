from typing import Callable, Dict, Optional, Mapping
import glob
from pathlib import Path

from catalyst import utils
from catalyst.contrib.data.cv.reader import ImageReader
from catalyst.data.dataset import PathsDataset
from catalyst.data.reader import ReaderCompose, ScalarReader


class ImageFolderDataset(PathsDataset):
    def __init__(
        self,
        rootpath: str,
        dir2class: Optional[Mapping[str, int]] = None,
        dict_transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        """Constructor method for the :class:`ImageFolderDataset` class.

        Args:
            rootpath (str):
            dir2class (Mapping[str, int], optional):
            dict_transform (Callable[[Dict], Dict]], optional):
        """
        files = glob.iglob(f"{rootpath}/**/*")
        images = sorted(filter(utils.has_image_extension, files))

        if dir2class is None:
            dirs = sorted({Path(f).parent.name for f in images})
            dir2class = {dirname: index for index, dirname in enumerate(dirs)}

        super().__init__(
            filenames=images,
            open_fn=ReaderCompose(
                [
                    ImageReader(input_key="image", rootpath=rootpath),
                    ScalarReader(
                        input_key="targets",
                        output_key="targets",
                        dtype=int,
                        default_value=-1,
                    ),
                ]
            ),
            label_fn=lambda fn: dir2class[Path(fn).parent.name],
            features_key="image",
            dict_transform=dict_transform,
        )


__all__ = ["ImageFolderDataset"]
