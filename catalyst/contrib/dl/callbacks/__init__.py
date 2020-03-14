# flake8: noqa
import logging
import os

from .criterion import CriterionAggregatorCallback
from .cutmix_callback import CutmixCallback
from .knn import KNNMetricCallback
from .telegram_logger import TelegramLogger

logger = logging.getLogger(__name__)

try:
    import alchemy
    from .alchemy import AlchemyLogger
except ImportError as ex:
    logger.warning(
        "alchemy not available, to install alchemy, "
        "run `pip install alchemy-catalyst`."
    )
    if os.environ.get("USE_ALCHEMY", "0") == "1":
        raise ex

try:
    import neptune
    from .neptune import NeptuneLogger
except ImportError as ex:
    if os.environ.get("USE_NEPTUNE", "0") == "1":
        logger.warning(
            "neptune not available, to install neptune, "
            "run `pip install neptune-client`."
        )
        raise ex
