# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.core.callback import (
    ICallback,
    Callback,
    CallbackList,
    CallbackWrapper,
    CallbackScope,
    CallbackNode,
    CallbackOrder,
)

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.batch_transform import BatchTransformCallback
from catalyst.callbacks.checkpoint import ICheckpointCallback, CheckpointCallback
from catalyst.callbacks.control_flow import ControlFlowCallback
from catalyst.callbacks.criterion import ICriterionCallback, CriterionCallback
from catalyst.callbacks.metric import (
    BatchMetricCallback,
    IMetricCallback,
    LoaderMetricCallback,
    FunctionalBatchMetricCallback,
)
from catalyst.callbacks.metric_aggregation import MetricAggregationCallback
from catalyst.callbacks.misc import (
    TimerCallback,
    TqdmCallback,
    CheckRunCallback,
    IBatchMetricHandlerCallback,
    IEpochMetricHandlerCallback,
    EarlyStoppingCallback,
)
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback

if SETTINGS.onnx_required:
    from catalyst.callbacks.onnx import OnnxCallback

if SETTINGS.optuna_required:
    from catalyst.callbacks.optuna import OptunaPruningCallback

from catalyst.callbacks.periodic_loader import PeriodicLoaderCallback

if SETTINGS.pruning_required:
    from catalyst.callbacks.pruning import PruningCallback

if SETTINGS.quantization_required:
    from catalyst.callbacks.quantization import QuantizationCallback

from catalyst.callbacks.scheduler import (
    ISchedulerCallback,
    SchedulerCallback,
    ILRUpdater,
    LRFinder,
)

from catalyst.callbacks.tracing import TracingCallback

from catalyst.callbacks.metrics import *
