# flake8: noqa
import logging

from catalyst.contrib.datasets.mnist import (
    MNIST,
    MnistMLDataset,
    MnistQGDataset,
)
from catalyst.tools import settings

logger = logging.getLogger(__name__)

try:
    from catalyst.contrib.datasets.cv import *
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
