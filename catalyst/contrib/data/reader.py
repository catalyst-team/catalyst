from typing import Callable, List, Optional, Type
import functools

import numpy as np

from catalyst.utils.numpy import get_one_hot


class IReader:
    """Reader abstraction for all Readers.

    Applies a function to an element of your data.
    For example to a row from csv, or to an image, etc.

    All inherited classes have to implement `__call__`.
    """

    def __init__(self, input_key: str, output_key: str):
        """
        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result,
                default: ``input_key``
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
        raise NotImplementedError("You cannot apply a transformation using `BaseReader`")


class ScalarReader(IReader):
    """
    Numeric data reader abstraction.
    Reads a single float, int, str or other from data
    """

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        dtype: Type = np.float32,
        default_value: float = None,
        one_hot_classes: int = None,
        smoothing: float = None,
    ):
        """
        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result,
                default: ``input_key``
            dtype: datatype of scalar values to use
            default_value: default value to use if something goes wrong
            one_hot_classes: number of one-hot classes
            smoothing (float, optional): if specified applies label smoothing
                to one_hot classes
        """
        super().__init__(input_key, output_key or input_key)
        self.dtype = dtype
        self.default_value = default_value
        self.one_hot_classes = one_hot_classes
        self.smoothing = smoothing
        if self.one_hot_classes is not None and self.smoothing is not None:
            assert 0.0 < smoothing < 1.0, (
                "If smoothing is specified it must be in (0; 1), " + f"got {smoothing}"
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
            scalar = get_one_hot(scalar, self.one_hot_classes, smoothing=self.smoothing)
        output = {self.output_key: scalar}
        return output


class LambdaReader(IReader):
    """
    Reader abstraction with an lambda encoders.
    Can read an elem from dataset and apply `encode_fn` function to it.
    """

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        lambda_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result
            lambda_fn: encode function to use to prepare your data
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

    def __init__(self, transforms: List[IReader]):
        """
        Args:
            transforms: list of reader to compose
            mixins: list of mixins to use
        """
        self.transforms = transforms

    def __call__(self, element):
        """
        Reads a row from your annotations dict and applies all readers and mixins

        Args:
            element: elem in your dataset.

        Returns:
            Value after applying all readers and mixins
        """
        result = {}
        for transform_fn in self.transforms:
            result = {**result, **transform_fn(element)}
        return result


__all__ = [
    "IReader",
    "ScalarReader",
    "LambdaReader",
    "ReaderCompose",
]
