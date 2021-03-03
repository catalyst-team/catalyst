# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import ICheckpointCallback, CheckpointCallback
from catalyst.callbacks.control_flow import ControlFlowCallback
from catalyst.callbacks.criterion import ICriterionCallback, CriterionCallback
from catalyst.callbacks.metric import IMetricCallback, BatchMetricCallback, LoaderMetricCallback
from catalyst.callbacks.misc import (
    TimerCallback,
    TqdmCallback,
    CheckRunCallback,
    IBatchMetricHandlerCallback,
    IEpochMetricHandlerCallback,
    EarlyStoppingCallback,
)
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.periodic_loader import PeriodicLoaderCallback
from catalyst.callbacks.scheduler import (
    ISchedulerCallback,
    SchedulerCallback,
    ILRUpdater,
    LRFinder,
)

# if SETTINGS.use_quantization:
#     from catalyst.callbacks.quantization import DynamicQuantizationCallback

if SETTINGS.use_pruning:
    from catalyst.callbacks.pruning import PruningCallback

if SETTINGS.use_optuna:
    from catalyst.callbacks.optuna import OptunaPruningCallback

from catalyst.callbacks.metrics import *
