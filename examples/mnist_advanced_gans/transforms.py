from typing import Tuple, Union

import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

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

    def add_targets(self, additional_targets):
        pass  # compatibility with albumentations


class AdditionalNoiseTensor(AdditionalValue):

    def __init__(self,
                 tensor_size: Tuple[int, ...],
                 output_key: str = None):
        super().__init__(output_key)
        self.tensor_size = tensor_size

    def _compute_output(self, dict_):
        return torch.randn(self.tensor_size)


class AdditionalScalar(AdditionalValue):

    def __init__(self,
                 value: Union[int, float],
                 output_key: str = None):
        super().__init__(output_key)
        self.value = value

    def _compute_output(self, dict_):
        return torch.tensor([self.value])
        # return self.value


registry.Transform(ToTensorV2)
registry.Transform(AsImage)

registry.Transform(AdditionalNoiseTensor)
registry.Transform(AdditionalScalar)
