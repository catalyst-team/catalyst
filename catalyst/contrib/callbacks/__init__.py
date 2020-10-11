# flake8: noqa
import logging

from torch.jit.frontend import UnsupportedNodeError

from catalyst.settings import SETTINGS

from catalyst.contrib.callbacks.confusion_matrix_logger import (
    ConfusionMatrixCallback,
)
from catalyst.contrib.callbacks.cutmix_callback import CutmixCallback
from catalyst.contrib.callbacks.gradnorm_logger import GradNormLogger
from catalyst.contrib.callbacks.inference_callback import InferCallback
from catalyst.contrib.callbacks.knn_metric import KNNMetricCallback
from catalyst.contrib.callbacks.mixup_callback import MixupCallback
from catalyst.contrib.callbacks.perplexity_metric import (
    PerplexityMetricCallback,
)
from catalyst.contrib.callbacks.telegram_logger import TelegramLogger

logger = logging.getLogger(__name__)

try:
    import imageio
    from catalyst.contrib.callbacks.mask_inference import InferMaskCallback
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex

try:
    import kornia
    from catalyst.contrib.callbacks.kornia_transform import (
        BatchTransformCallback,
    )
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
except UnsupportedNodeError as ex:
    logger.warning(
        "kornia has requirement torch>=1.5.0,"
        " probably you have an old version of torch which is incompatible.\n"
        "To update pytorch, run `pip install -U 'torch>=1.5.0'`."
    )
    if SETTINGS.kornia_required:
        raise ex

try:
    import alchemy
    from catalyst.contrib.callbacks.alchemy_logger import AlchemyLogger
except ImportError as ex:
    if SETTINGS.alchemy_logger_required:
        logger.warning(
            "alchemy not available, to install alchemy, "
            "run `pip install alchemy`."
        )
        raise ex


try:
    import neptune
    from catalyst.contrib.callbacks.neptune_logger import NeptuneLogger
except ImportError as ex:
    if SETTINGS.neptune_logger_required:
        logger.warning(
            "neptune not available, to install neptune, "
            "run `pip install neptune-client`."
        )
        raise ex

try:
    import wandb
    from catalyst.contrib.callbacks.wandb_logger import WandbLogger
except ImportError as ex:
    if SETTINGS.wandb_logger_required:
        logger.warning(
            "wandb not available, to install wandb, "
            "run `pip install wandb`."
        )
        raise ex

try:
    import optuna
    from catalyst.contrib.callbacks.optuna_callback import (
        OptunaPruningCallback,
        OptunaCallback,
    )
except ImportError as ex:
    if SETTINGS.optuna_required:
        logger.warning(
            "optuna not available, to install optuna, "
            "run `pip install optuna`."
        )
