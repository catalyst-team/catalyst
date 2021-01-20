# flake8: noqa
import logging

from torch.jit.frontend import UnsupportedNodeError

from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

# alchemy logger
# try:
#     import alchemy
#     from catalyst.contrib.callbacks.alchemy_logger import AlchemyLogger
# except ImportError as ex:
#     if SETTINGS.alchemy_required:
#         logger.warning(
#             "alchemy not available, to install alchemy, "
#             "run `pip install alchemy`."
#         )
#         raise ex

# confusion matrix logger
# try:
#     import matplotlib  # noqa: F401
#
#     from catalyst.contrib.callbacks.confusion_matrix_logger import (
#         ConfusionMatrixCallback,
#     )
# except ModuleNotFoundError as ex:
#     if SETTINGS.matplotlib_required:
#         logger.warning(
#             "matplotlib is not available, to install matplotlib,"
#             " run `pip install matplotlib`."
#         )
#         raise ex
# except ImportError as ex:
#     if SETTINGS.matplotlib_required:
#         logger.warning(
#             "matplotlib is not available, to install matplotlib,"
#             " run `pip install matplotlib`."
#         )
#         raise ex

# from catalyst.contrib.callbacks.cutmix_callback import CutmixCallback
# from catalyst.contrib.callbacks.gradnorm_logger import GradNormLogger
# from catalyst.contrib.callbacks.inference_callback import InferCallback

# kornia
# try:
#     import kornia
#     from catalyst.contrib.callbacks.kornia_transform import (
#         BatchTransformCallback,
#     )
# except ImportError as ex:
#     if SETTINGS.cv_required:
#         logger.warning(
#             "some of catalyst-cv dependencies are not available,"
#             " to install dependencies, run `pip install catalyst[cv]`."
#         )
#         raise ex
# except UnsupportedNodeError as ex:
#     logger.warning(
#         "kornia has requirement torch>=1.6.0,"
#         " probably you have an old version of torch which is incompatible.\n"
#         "To update pytorch, run `pip install -U 'torch>=1.6.0'`."
#     )
#     if SETTINGS.kornia_required:
#         raise ex


# try:
#     import imageio
#     from catalyst.contrib.callbacks.mask_inference import InferMaskCallback
#     from catalyst.contrib.callbacks.draw_masks_callback import (
#         DrawMasksCallback,
#     )
# except ModuleNotFoundError as ex:
#     if SETTINGS.cv_required:
#         logger.warning(
#             "some of catalyst-cv dependencies are not available,"
#             " to install dependencies, run `pip install catalyst[cv]`."
#         )
#         raise ex
# except ImportError as ex:
#     if SETTINGS.cv_required:
#         logger.warning(
#             "some of catalyst-cv dependencies are not available,"
#             " to install dependencies, run `pip install catalyst[cv]`."
#         )
#         raise ex

# from catalyst.contrib.callbacks.mixup_callback import MixupCallback

# try:
#     import neptune
#     from catalyst.contrib.callbacks.neptune_logger import NeptuneLogger
# except ModuleNotFoundError as ex:
#     if SETTINGS.neptune_required:
#         logger.warning(
#             "neptune not available, to install neptune, "
#             "run `pip install neptune-client`."
#         )
#         raise ex
# except ImportError as ex:
#     if SETTINGS.neptune_required:
#         logger.warning(
#             "neptune not available, to install neptune, "
#             "run `pip install neptune-client`."
#         )
#         raise ex

try:
    import optuna
    from catalyst.contrib.callbacks.optuna_callback import OptunaPruningCallback
except ModuleNotFoundError as ex:
    if SETTINGS.optuna_required:
        logger.warning("optuna not available, to install optuna, " "run `pip install optuna`.")
        raise ex
except ImportError as ex:
    if SETTINGS.optuna_required:
        logger.warning("optuna not available, to install optuna, " "run `pip install optuna`.")
        raise ex

# from catalyst.contrib.callbacks.telegram_logger import TelegramLogger

# try:
#     import wandb
#     from catalyst.contrib.callbacks.wandb_logger import WandbLogger
# except ModuleNotFoundError as ex:
#     if SETTINGS.wandb_required:
#         logger.warning(
#             "wandb not available, to install wandb, "
#             "run `pip install wandb`."
#         )
#         raise ex
# except ImportError as ex:
#     if SETTINGS.wandb_required:
#         logger.warning(
#             "wandb not available, to install wandb, "
#             "run `pip install wandb`."
#         )
#         raise ex
