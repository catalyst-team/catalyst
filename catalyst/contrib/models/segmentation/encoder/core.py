from typing import List  # isort:skip
from abc import ABC, abstractmethod

import torch.nn as nn


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


class EncoderSpec(ABC, nn.Module):
    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def out_strides(self) -> List[int]:
        pass
