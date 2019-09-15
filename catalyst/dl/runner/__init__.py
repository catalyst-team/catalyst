# flake8: noqa
import logging
import os

logger = logging.getLogger(__name__)

from .supervised import SupervisedRunner

try:
    import wandb
    from .wandb import WandbRunner, SupervisedWandbRunner
except ImportError:
    if os.environ.get("USE_WANDB", "0") == "1":
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
