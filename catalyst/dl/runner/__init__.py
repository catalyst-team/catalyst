# flake8: noqa
import logging
import os

from .gan import GanRunner
from .supervised import SupervisedRunner

logger = logging.getLogger(__name__)


try:
    import wandb
    from .wandb import WandbRunner, SupervisedWandbRunner
except ImportError as ex:
    if os.environ.get("USE_WANDB", "0") == "1":
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
        raise ex
