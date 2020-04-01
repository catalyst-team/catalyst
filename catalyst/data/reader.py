from typing import Callable, List, Tuple, Type, Union
import functools

import numpy as np

from catalyst.utils import get_one_hot, imread, mimread


class ReaderSpec:
    """Reader abstraction for all Readers.

    Applies a function to an element of your data.
    For example to a row from csv, or to an image, etc.

    All inherited classes have to implement `__call__`.
    """

    def __init__(self, input_key: str, output_key: str):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, element):
        """
        Reads a row from your annotations dict and transfer it to data,
        needed by your network for example open image by path,
        or read string and tokenize it.

        Args:
            element: elem in your dataset

        Returns:
            Data object used for your neural network
        """
        raise NotImplementedError(
            "You cannot apply a transformation using `BaseReader`"
        )


class ImageReader(ReaderSpec):
    """Image reader abstraction. Reads images from a ``csv`` dataset."""

    def __init__(
        self,
        input_key: str,
        output_key: str,
        rootpath: str = None,
        grayscale: bool = False,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            grayscale (bool): flag if you need to work only
                with grayscale images
        """
        super().__init__(input_key, output_key)
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
        img = imread(
            image_name, rootpath=self.rootpath, grayscale=self.grayscale
        )

        output = {self.output_key: img}
        return output


class MaskReader(ReaderSpec):
    """Mask reader abstraction. Reads masks from a `csv` dataset."""

    def __init__(
        self,
        input_key: str,
        output_key: str,
        rootpath: str = None,
        clip_range: Tuple[Union[int, float], Union[int, float]] = (0, 1),
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            clip_range (Tuple[int, int]): lower and upper interval edges,
                image values outside the interval are clipped
                to the interval edges
        """
        super().__init__(input_key, output_key)
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
        mask = mimread(mask_name, rootpath=self.rootpath, clip_range=self.clip)

        output = {self.output_key: mask}
        return output


class ScalarReader(ReaderSpec):
    """
    Numeric data reader abstraction.
    Reads a single float, int, str or other from data
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        dtype: Type = np.float32,
        default_value: float = None,
        one_hot_classes: int = None,
        smoothing: float = None,
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
            dtype (type): datatype of scalar values to use
            default_value: default value to use if something goes wrong
            one_hot_classes (int): number of one-hot classes
            smoothing (float, optional): if specified applies label smoothing
                to one_hot classes
        """
        super().__init__(input_key, output_key)
        self.dtype = dtype
        self.default_value = default_value
        self.one_hot_classes = one_hot_classes
        self.smoothing = smoothing
        if self.one_hot_classes is not None and self.smoothing is not None:
            assert 0.0 < smoothing < 1.0, (
                f"If smoothing is specified it must be in (0; 1), "
                f"got {smoothing}"
            )

    def __call__(self, element):
        """
        Reads a row from your annotations dict and
        transfer it to a single value

        Args:
            element: elem in your dataset

        Returns:
            dtype: Scalar value
        """
        scalar = self.dtype(element.get(self.input_key, self.default_value))
        if self.one_hot_classes is not None:
            scalar = get_one_hot(
                scalar, self.one_hot_classes, smoothing=self.smoothing
            )
        output = {self.output_key: scalar}
        return output


class LambdaReader(ReaderSpec):
    """
    Reader abstraction with an lambda encoders.
    Can read an elem from dataset and apply `encode_fn` function to it.
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        lambda_fn: Callable = lambda x: x,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
            lambda_fn (callable): encode function to use to prepare your data
                (for example convert chars/words/tokens to indices, etc)
            kwargs: kwargs for encode function
        """
        super().__init__(input_key, output_key)
        self.lambda_fn = functools.partial(lambda_fn, **kwargs)

    def __call__(self, element):
        """
        Reads a row from your annotations dict
        and applies `encode_fn` function.

        Args:
            element: elem in your dataset.

        Returns:
            Value after applying `lambda_fn` function
        """
        if self.input_key is not None:
            element = element[self.input_key]
        output = self.lambda_fn(element)
        if self.output_key is not None:
            output = {self.output_key: output}
        return output


class ReaderCompose(object):
    """Abstraction to compose several readers into one open function."""

    def __init__(self, readers: List[ReaderSpec], mixins: list = None):
        """
        Args:
            readers (List[ReaderSpec]): list of reader to compose
            mixins (list): list of mixins to use
        """
        self.readers = readers
        self.mixins = mixins or []

    def __call__(self, element):
        """
        Reads a row from your annotations dict
        and applies all readers and mixins

        Args:
            element: elem in your dataset.

        Returns:
            Value after applying all readers and mixins
        """
        result = {}
        for fn in self.readers:
            result = {**result, **fn(element)}
        for fn in self.mixins:
            result = {**result, **fn(result)}
        return result


__all__ = [
    "ReaderSpec",
    "ImageReader",
    "ScalarReader",
    "LambdaReader",
    "ReaderCompose",
]
