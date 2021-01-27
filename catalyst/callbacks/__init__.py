# flake8: noqa

from catalyst.settings import (
    IS_QUANTIZATION_AVAILABLE,
    IS_PRUNING_AVAILABLE,
)

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import ICheckpointCallback, CheckpointCallback
from catalyst.callbacks.confusion_matrix import ConfusionMatrixCallback
from catalyst.callbacks.control_flow import ControlFlowCallback
from catalyst.callbacks.criterion import ICriterionCallback, CriterionCallback
from catalyst.callbacks.metric import IMetricCallback, MetricCallback, LoaderMetricCallback
from catalyst.callbacks.misc import (
    TimerCallback,
    VerboseCallback,
    CheckRunCallback,
    MetricHandler,
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

# if IS_QUANTIZATION_AVAILABLE:
#     from catalyst.callbacks.quantization import DynamicQuantizationCallback

# if IS_PRUNING_AVAILABLE:
#     from catalyst.callbacks.pruning import PruningCallback

from catalyst.contrib.callbacks.optuna_callback import OptunaPruningCallback

# from catalyst.contrib.callbacks import *
