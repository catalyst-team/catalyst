# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.models.functional import (
    get_convolution_net,
    get_linear_net,
)
from catalyst.contrib.models.hydra import Hydra
from catalyst.contrib.models.sequential import (
    ResidualWrapper,
    SequentialNet,
)
from catalyst.contrib.models.mnist import MnistSimpleNet


if SETTINGS.cv_required:
    from catalyst.contrib.models.cv import *
