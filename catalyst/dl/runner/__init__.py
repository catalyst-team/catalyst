# flake8: noqa
import logging
import os

logger = logging.getLogger(__name__)

from .supervised import SupervisedRunner
from .gan import GanRunner

if os.environ.get("USE_WANDB", "1") == "1":
    try:
        import wandb
        from .wandb import WandbRunner, SupervisedWandbRunner
    except ImportError:
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
