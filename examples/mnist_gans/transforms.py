# flake8: noqa
from typing import Tuple, Union

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform
import torch

from catalyst.dl import registry


class AsImage(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if img.ndim == 2:
            return np.expand_dims(img, axis=-1)
        return img

    def get_transform_init_args_names(self):
        return []


class AdditionalValue:
    def __init__(self, output_key: str = None, **kwargs):
        self.output_key = output_key

    def __call__(self, force_apply=False, **dict_):
        assert self.output_key not in dict_, \
            "Output key is supposed not to be present in dict"
        dict_[self.output_key] = self._compute_output(dict_)
        return dict_

    def _compute_output(self, dict_):
        raise NotImplementedError()


class AdditionalNoiseTensor(AdditionalValue):
    def __init__(self, tensor_size: Tuple[int, ...], output_key: str = None):
        super().__init__(output_key)
        self.tensor_size = tensor_size

    def _compute_output(self, dict_):
        return torch.randn(self.tensor_size)


class AdditionalScalar(AdditionalValue):
    def __init__(self, value: Union[int, float], output_key: str = None):
        super().__init__(output_key)
        self.value = value

    def _compute_output(self, dict_):
        return torch.tensor([self.value])


class OneHotTargetTransform:
    """Adds one-hot encoded target to input dict
    i.e. dict_[output_key] = one_hot_encode(dict_[input_key])
    """
    def __init__(self, input_key: str, output_key: str, num_classes: int):
        self.input_key = input_key
        self.output_key = output_key
        self.num_classes = num_classes

    def __call__(self, force_apply=False, **dict_):
        class_id = dict_[self.input_key]
        assert self.output_key not in dict_, \
            "Output key is supposed not to be present in dict"

        target = np.zeros((self.num_classes, ), dtype=np.int64)
        target[class_id] = 1
        dict_[self.output_key] = target

        return dict_


registry.Transform(AsImage)

registry.Transform(AdditionalNoiseTensor)
registry.Transform(AdditionalScalar)

registry.Transform(OneHotTargetTransform)
