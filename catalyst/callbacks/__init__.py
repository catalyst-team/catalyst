# flake8: noqa

from catalyst.settings import (
    IS_QUANTIZATION_AVAILABLE,
    IS_PRUNING_AVAILABLE,
)

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import (
    ICheckpointCallback,
    BaseCheckpointCallback,
    CheckpointCallback,
    IterationCheckpointCallback,
)
from catalyst.callbacks.control_flow import ControlFlowCallback
from catalyst.callbacks.criterion import CriterionCallback
from catalyst.callbacks.early_stop import (
    EarlyStoppingCallback,
    CheckRunCallback,
)
from catalyst.callbacks.exception import ExceptionCallback
from catalyst.callbacks.logging import (
    ILoggerCallback,
    VerboseLogger,
    ConsoleLogger,
    TensorboardLogger,
)
from catalyst.callbacks.meter import MeterMetricsCallback
from catalyst.callbacks.metric import (
    IMetricCallback,
    IBatchMetricCallback,
    ILoaderMetricCallback,
    BatchMetricCallback,
    LoaderMetricCallback,
    MetricCallback,
    MetricAggregationCallback,
    MetricManagerCallback,
)
from catalyst.callbacks.optimizer import (
    IOptimizerCallback,
    OptimizerCallback,
    AMPOptimizerCallback,
)
from catalyst.callbacks.periodic_loader import PeriodicLoaderCallback
from catalyst.callbacks.scheduler import (
    ISchedulerCallback,
    ILRUpdater,
    SchedulerCallback,
    LRFinder,
)
from catalyst.callbacks.timer import TimerCallback
from catalyst.callbacks.tracing import TracingCallback, TracerCallback
from catalyst.callbacks.validation import ValidationManagerCallback

from catalyst.callbacks.metrics import *

if IS_QUANTIZATION_AVAILABLE:
    from catalyst.callbacks.quantization import DynamicQuantizationCallback

if IS_PRUNING_AVAILABLE:
    from catalyst.callbacks.pruning import PruningCallback

from catalyst.contrib.callbacks import *
