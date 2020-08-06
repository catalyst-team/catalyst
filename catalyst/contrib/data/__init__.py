# flake8: noqa
import logging

from catalyst.tools import settings

from catalyst.contrib.data.transforms import (
    Compose,
    Normalize,
    normalize,
    to_tensor,
    ToTensor,
)


logger = logging.getLogger(__name__)

try:
    from catalyst.contrib.data.cv import *
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex


try:
    from catalyst.contrib.data.nlp import *
except ImportError as ex:
    if settings.nlp_required:
        logger.warning(
            "some of catalyst-nlp dependencies not available,"
            " to install dependencies, run `pip install catalyst[nlp]`."
        )
        raise ex
