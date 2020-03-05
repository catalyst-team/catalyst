# flake8: noqa
import logging
import os
import warnings

logger = logging.getLogger(__name__)
warnings.simplefilter("default")

try:
    import alchemy
    from .alchemy import AlchemyRunner, SupervisedAlchemyRunner

    warnings.warn(
        "AlchemyRunner and SupervisedAlchemyRunner are deprecated; "
        "use AlchemyLogger instead (`from catalyst.dl import AlchemyLogger`)",
        DeprecationWarning
    )
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

    warnings.warn(
        "NeptuneRunner and SupervisedNeptuneRunner are deprecated; "
        "will be removed in 20.04 release", DeprecationWarning
    )
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

    warnings.warn(
        "WandbRunner and SupervisedWandbRunner are deprecated; "
        "will be removed in 20.04 release", DeprecationWarning
    )
except ImportError as ex:
    if os.environ.get("USE_WANDB", "0") == "1":
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
        raise ex
