# flake8: noqa
import logging

from catalyst.settings import SETTINGS

from catalyst.contrib.datasets.mnist import (
    MnistMLDataset,
    MnistQGDataset,
    MNIST,
)

from catalyst.contrib.datasets.movielens import MovieLens

logger = logging.getLogger(__name__)

try:
    from catalyst.contrib.datasets.cv import *
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
