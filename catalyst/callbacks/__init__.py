# flake8: noqa

from distutils.version import LooseVersion

import torch

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.batch_transform import BatchTransformCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.control_flow import ControlFlowCallback
from catalyst.callbacks.criterion import CriterionCallback, ICriterionCallback
from catalyst.callbacks.metric import (
    BatchMetricCallback,
    FunctionalBatchMetricCallback,
    IMetricCallback,
    LoaderMetricCallback,
)
from catalyst.callbacks.metric_aggregation import MetricAggregationCallback
from catalyst.callbacks.misc import (
    CheckRunCallback,
    EarlyStoppingCallback,
    IBatchMetricHandlerCallback,
    IEpochMetricHandlerCallback,
    TimerCallback,
    TqdmCallback,
)
from catalyst.callbacks.mixup import MixupCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.core.callback import (
    Callback,
    CallbackList,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
    CallbackWrapper,
    ICallback,
)
from catalyst.settings import SETTINGS

if SETTINGS.onnx_required:
    from catalyst.callbacks.onnx import OnnxCallback

if SETTINGS.optuna_required:
    from catalyst.callbacks.optuna import OptunaPruningCallback

from catalyst.callbacks.periodic_loader import PeriodicLoaderCallback

if SETTINGS.pruning_required:
    from catalyst.callbacks.pruning import PruningCallback

if SETTINGS.quantization_required:
    from catalyst.callbacks.quantization import QuantizationCallback

if LooseVersion(torch.__version__) >= LooseVersion("1.8.1"):
    from catalyst.callbacks.profiler import ProfilerCallback

from catalyst.callbacks.metrics import *
from catalyst.callbacks.scheduler import (
    ILRUpdater,
    ISchedulerCallback,
    LRFinder,
    SchedulerCallback,
)

if SETTINGS.ml_required:
    from catalyst.callbacks.sklearn_model import SklearnModelCallback

from catalyst.callbacks.tracing import TracingCallback
