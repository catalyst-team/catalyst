# flake8: noqa
import logging
import os

logger = logging.getLogger(__name__)

try:
    import alchemy
    from .alchemy import AlchemyRunner, SupervisedAlchemyRunner
except ImportError as ex:
    logger.warning(
        "alchemy not available, to install alchemy, "
        "run `pip install alchemy-catalyst`."
    )
    if os.environ.get("USE_ALCHEMY", "0") == "1":
        raise ex

try:
    import neptune
    from .neptune import NeptuneRunner, SupervisedNeptuneRunner
except ImportError as ex:
    if os.environ.get("USE_NEPTUNE", "0") == "1":
        logger.warning(
            "neptune not available, to install neptune, "
            "run `pip install neptune-client`."
        )
        raise ex

try:
    import wandb
    from .wandb import WandbRunner, SupervisedWandbRunner
except ImportError as ex:
    if os.environ.get("USE_WANDB", "0") == "1":
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
        raise ex
