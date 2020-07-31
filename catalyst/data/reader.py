from typing import Callable, List, Optional, Type
import functools

import numpy as np

from catalyst.utils import get_one_hot


class ReaderSpec:
    """Reader abstraction for all Readers.

    Applies a function to an element of your data.
    For example to a row from csv, or to an image, etc.

    All inherited classes have to implement `__call__`.
    """

    def __init__(self, input_key: str, output_key: Optional[str] = None):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (Optional[str]): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key or input_key

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


class ScalarReader(ReaderSpec):
    """
    Numeric data reader abstraction.
    Reads a single float, int, str or other from data
    """

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str],
        dtype: Type = np.float32,
        default_value: float = None,
        one_hot_classes: int = None,
        smoothing: float = None,
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (Optional[str]): output key to use to store the result
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
                "If smoothing is specified it must be in (0; 1), "
                + f"got {smoothing}"
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
        output_key: Optional[str],
        lambda_fn: Callable = None,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (Optional[str]): output key to use to store the result
            lambda_fn (callable): encode function to use to prepare your data
              (for example convert chars/words/tokens to indices, etc)
            kwargs: kwargs for encode function
        """
        super().__init__(input_key, output_key)
        lambda_fn = lambda_fn or (lambda x: x)
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
        for reader_fn in self.readers:
            result = {**result, **reader_fn(element)}
        for mixin_fn in self.mixins:
            result = {**result, **mixin_fn(result)}
        return result


try:
    # @TODO: remove hotfix
    from catalyst.contrib.data.cv.reader import (  # noqa: F401
        ImageReader,
        MaskReader,
    )

    __all__ = [
        "ReaderSpec",
        "ScalarReader",
        "LambdaReader",
        "ReaderCompose",
        "ImageReader",
        "MaskReader",
    ]
except ImportError:
    __all__ = [
        "ReaderSpec",
        "ScalarReader",
        "LambdaReader",
        "ReaderCompose",
    ]
