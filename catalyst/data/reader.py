from typing import Callable, Type, List

import numpy as np
from catalyst.data.functional import read_image


class BaseReader:
    """
    Reader abstraction for all Readers. All inherited classes has to implement `__call__`
    """

    def __init__(self, input_key: str, output_key: str):
        """
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, row):
        """
        Applies a row from your annotations dict and
            transfer it to data, needed by your network
            for example open image by path, or read string and tokenize it.
        :param row: elem in your dataset. It can be row in csv, or image for example.
        :return: Data object used for your neural network
        """
        raise NotImplementedError(
            "You cannot apply a transformation using `BaseReader`"
        )


class ImageReader(BaseReader):
    """
    Image reader abstraction.
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        datapath: str = None,
        grayscale: bool = False
    ):
        """
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        :param datapath: path to images dataset
            (so your can use relative paths in annotations)
        :param grayscale: boolean flag
            if you need to work only with grayscale images
        """
        super().__init__(input_key, output_key)
        self.datapath = datapath
        self.grayscale = grayscale

    def __call__(self, row):
        image_name = str(row[self.input_key])
        img = read_image(
            image_name, datapath=self.datapath, grayscale=self.grayscale
        )

        result = {self.output_key: img}
        return result


class ScalarReader(BaseReader):
    """
    Numeric data reader abstraction.
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
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        :param dtype: datatype of scalar values to use
        :param default_value: default value to use if something goes wrong
        """
        super().__init__(input_key, output_key)
        self.dtype = dtype
        self.default_value = default_value
        self.one_hot_classes = one_hot_classes

    def __call__(self, row):
        scalar = self.dtype(row.get(self.input_key, self.default_value))
        if self.one_hot_classes is not None \
                and scalar is not None and scalar >= 0:
            one_hot = np.zeros(self.one_hot_classes, dtype=np.float32)
            one_hot[scalar] = 1.0
            scalar = one_hot
        result = {self.output_key: scalar}
        return result


class TextReader(BaseReader):
    """
    Text reader abstraction.
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        encode_fn: Callable = lambda x: x
    ):
        """
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        :param encode_fn: encode function to use to prepare your data
            for example convert chars/words/tokens to indices, etc
        """
        super().__init__(input_key, output_key)
        self.encode_fn = encode_fn

    def __call__(self, row):
        text = row[self.input_key]
        text = self.encode_fn(text)
        result = {self.output_key: text}
        return result


class ReaderCompose(object):
    """
    Abstraction to compose several readers into one open function.
    """

    def __init__(self, readers: List[BaseReader], mixins: [] = None):
        """
        :param readers: list of reader to compose
        :param mixins: list of mixins to use
        """
        self.readers = readers
        self.mixins = mixins or []

    def __call__(self, row):
        result = {}
        for fn in self.readers:
            result = {**result, **fn(row)}
        for fn in self.mixins:
            result = {**result, **fn(result)}
        return result
