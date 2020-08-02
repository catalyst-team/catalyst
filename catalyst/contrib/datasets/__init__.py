# flake8: noqa

import logging

from catalyst.tools import settings

from catalyst.contrib.datasets.mnist import MNIST
from catalyst.contrib.datasets.mnist_qg import MnistQGDataset

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
