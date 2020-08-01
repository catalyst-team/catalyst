# flake8: noqa
import logging

from catalyst.tools import settings

from catalyst.core.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.core.callbacks.checkpoint import (
    CheckpointCallback,
    IterationCheckpointCallback,
)

logger = logging.getLogger(__name__)

try:
    from .pruning import PruningCallback
except ImportError as ex:
    logger.warning(
        "Quantization and pruning are not available,"
        "run `pip install torch>=1.4` to enable them."
    )
    if settings.pytorch_14:
        raise ex

from catalyst.core.callbacks.criterion import CriterionCallback
from catalyst.core.callbacks.early_stop import (
    CheckRunCallback,
    EarlyStoppingCallback,
)
from catalyst.core.callbacks.exception import ExceptionCallback
from catalyst.core.callbacks.logging import (
    ConsoleLogger,
    TensorboardLogger,
    VerboseLogger,
)
from catalyst.core.callbacks.metrics import *
from catalyst.core.callbacks.optimizer import OptimizerCallback
from catalyst.core.callbacks.scheduler import LRUpdater, SchedulerCallback
from catalyst.core.callbacks.timer import TimerCallback
from catalyst.core.callbacks.validation import ValidationManagerCallback
from catalyst.core.callbacks.control_flow import ControlFlowCallback
