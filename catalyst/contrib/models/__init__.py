# flake8: noqa
import logging

from catalyst.tools import settings

from catalyst.contrib.models.functional import (
    get_convolution_net,
    get_linear_net,
)
from catalyst.contrib.models.hydra import Hydra
from catalyst.contrib.models.sequential import (
    ResidualWrapper,
    SequentialNet,
)
from catalyst.contrib.models.simple_conv import SimpleConv

logger = logging.getLogger(__name__)

try:
    from catalyst.contrib.models.cv import *
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex


try:
    from catalyst.contrib.models.nlp import *
except ImportError as ex:
    if settings.nlp_required:
        logger.warning(
            "some of catalyst-nlp dependencies not available,"
            " to install dependencies, run `pip install catalyst[nlp]`."
        )
        raise ex
