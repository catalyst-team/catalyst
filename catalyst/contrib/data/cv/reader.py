from typing import Optional, Tuple, Union

from catalyst import utils
from catalyst.data.reader import ReaderSpec


class ImageReader(ReaderSpec):
    """Image reader abstraction. Reads images from a ``csv`` dataset."""

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
        grayscale: bool = False,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (Optional[str]): key to use to store the result,
                default: ``input_key``
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            grayscale (bool): flag if you need to work only
                with grayscale images
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.grayscale = grayscale

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image

        Args:
            element: elem in your dataset

        Returns:
            np.ndarray: Image
        """
        image_name = str(element[self.input_key])
        img = utils.imread(
            image_name, rootpath=self.rootpath, grayscale=self.grayscale
        )

        output = {self.output_key: img}
        return output


class MaskReader(ReaderSpec):
    """Mask reader abstraction. Reads masks from a `csv` dataset."""

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
        clip_range: Tuple[Union[int, float], Union[int, float]] = (0, 1),
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (Optional[str]): key to use to store the result,
                default: ``input_key``
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            clip_range (Tuple[int, int]): lower and upper interval edges,
                image values outside the interval are clipped
                to the interval edges
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.clip = clip_range

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to a mask

        Args:
            element: elem in your dataset.

        Returns:
            np.ndarray: Mask
        """
        mask_name = str(element[self.input_key])
        mask = utils.mimread(
            mask_name, rootpath=self.rootpath, clip_range=self.clip
        )

        output = {self.output_key: mask}
        return output


__all__ = ["ImageReader", "MaskReader"]
