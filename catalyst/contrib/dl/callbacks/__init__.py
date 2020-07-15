# flake8: noqa
import logging

from catalyst.tools import settings

from .cutmix_callback import CutmixCallback
from .gradnorm_logger import GradNormLogger
from .knn_metric import KNNMetricCallback
from .periodic_loader_callback import PeriodicLoaderCallback
from .perplexity_metric import PerplexityMetricCallback
from .telegram_logger import TelegramLogger
from .tracer_callback import TracerCallback

logger = logging.getLogger(__name__)

try:
    import imageio
    from .mask_inference import InferMaskCallback
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex

try:
    import alchemy
    from .alchemy_logger import AlchemyLogger
except ImportError as ex:
    if settings.alchemy_logger_required:
        logger.warning(
            "alchemy not available, to install alchemy, "
            "run `pip install alchemy`."
        )
        raise ex

try:
    import visdom
    from .visdom_logger import VisdomLogger
except ImportError as ex:
    if settings.visdom_logger_required:
        logger.warning(
            "visdom not available, to install visdom, "
            "run `pip install visdom`."
        )
        raise ex

try:
    import neptune
    from .neptune_logger import NeptuneLogger
except ImportError as ex:
    if settings.neptune_logger_required:
        logger.warning(
            "neptune not available, to install neptune, "
            "run `pip install neptune-client`."
        )
        raise ex

try:
    import wandb
    from .wandb_logger import WandbLogger
except ImportError as ex:
    if settings.wandb_logger_required:
        logger.warning(
            "wandb not available, to install wandb, "
            "run `pip install wandb`."
        )
        raise ex
