import functools
from typing import Callable, Type, List

import numpy as np
from catalyst.utils.image import imread


class ReaderSpec:
    """Reader abstraction for all Readers. Applies a function
    to an element of your data.
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

    def __call__(self, row):
        """Reads a row from your annotations dict and
        transfer it to data, needed by your network
        for example open image by path, or read string and tokenize it.

        Args:
            row: elem in your dataset.

        Returns:
            Data object used for your neural network
        """
        raise NotImplementedError(
            "You cannot apply a transformation using `BaseReader`"
        )


class ImageReader(ReaderSpec):
    """
    Image reader abstraction. Reads images from a `csv` dataset.
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        datapath: str = None,
        grayscale: bool = False
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            datapath (str): path to images dataset
                (so your can use relative paths in annotations)
            grayscale (bool): flag if you need to work only
                with grayscale images
        """
        super().__init__(input_key, output_key)
        self.datapath = datapath
        self.grayscale = grayscale

    def __call__(self, row):
        """Reads a row from your annotations dict with filename and
        transfer it to an image

        Args:
            row: elem in your dataset.

        Returns:
            np.ndarray: Image
        """
        image_name = str(row[self.input_key])
        img = imread(
            image_name, rootpath=self.datapath, grayscale=self.grayscale
        )

        result = {self.output_key: img}
        return result


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
        one_hot_classes: int = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
            dtype (type): datatype of scalar values to use
            default_value: default value to use if something goes wrong
            one_hot_classes (int): number of one-hot classes
        """
        super().__init__(input_key, output_key)
        self.dtype = dtype
        self.default_value = default_value
        self.one_hot_classes = one_hot_classes

    def __call__(self, row):
        """Reads a row from your annotations dict with filename and
        transfer it to a single value

        Args:
            row: elem in your dataset.

        Returns:
            dtype: Scalar value
        """
        scalar = self.dtype(row.get(self.input_key, self.default_value))
        if self.one_hot_classes is not None \
                and scalar is not None and scalar >= 0:
            one_hot = np.zeros(self.one_hot_classes, dtype=np.float32)
            one_hot[scalar] = 1.0
            scalar = one_hot
        result = {self.output_key: scalar}
        return result


class LambdaReader(ReaderSpec):
    """
    Reader abstraction with an lambda encoder.
    Can read an elem from dataset and apply `encode_fn` function to it
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        encode_fn: Callable = lambda x: x,
        **kwargs
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
            encode_fn (callable): encode function to use to prepare your data
                (for example convert chars/words/tokens to indices, etc)
            kwargs: kwargs for encode function
        """
        super().__init__(input_key, output_key)
        self.encode_fn = functools.partial(encode_fn, **kwargs)

    def __call__(self, row):
        """Reads a row from your annotations dict
        and applies `encode_fn` function

        Args:
            row: elem in your dataset.

        Returns:
            Value after applying `encode_fn` function
        """
        elem = row[self.input_key]
        elem = self.encode_fn(elem)
        result = {self.output_key: elem}
        return result


class ReaderCompose(object):
    """
    Abstraction to compose several readers into one open function.
    """

    def __init__(self, readers: List[ReaderSpec], mixins: [] = None):
        """
        Args:
            readers (List[ReaderSpec]): list of reader to compose
            mixins: list of mixins to use
        """
        self.readers = readers
        self.mixins = mixins or []

    def __call__(self, row):
        """Reads a row from your annotations dict
        and applies all readers and mixins

        Args:
            row: elem in your dataset.

        Returns:
            Value after applying all readers and mixins
        """
        result = {}
        for fn in self.readers:
            result = {**result, **fn(row)}
        for fn in self.mixins:
            result = {**result, **fn(result)}
        return result


__all__ = [
    "ReaderSpec", "ImageReader", "ScalarReader", "LambdaReader",
    "ReaderCompose"
]
