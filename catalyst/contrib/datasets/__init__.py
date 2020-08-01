# flake8: noqa
import logging

from catalyst.tools import settings

from .metric_learning import MetricLearningTrainDataset, QueryGalleryDataset
from .mnist import MnistMLDataset, MnistQGDataset, MNIST

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
