# flake8: noqa
import logging
import os

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
