# flake8: noqa
import logging
import os

logger = logging.getLogger(__name__)


try:
    import alchemy
    from .alchemy import AlchemyRunner, SupervisedAlchemyRunner
except ImportError as ex:
    if os.environ.get("USE_ALCHEMY", "0") == "1":
        logger.warning(
            "alchemy not available, to install alchemy, "
            "run `pip install alchemy-catalyst`."
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
