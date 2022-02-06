# flake8: noqa
from catalyst.settings import SETTINGS
from distutils.version import LooseVersion
import torch


from catalyst.core.callback import (
    Callback,
    CallbackOrder,
    CallbackWrapper,
    ICallback,
    IBackwardCallback,
    ICriterionCallback,
    IMetricCallback,
    IOptimizerCallback,
    ISchedulerCallback,
    ICheckpointCallback,
)


from catalyst.callbacks.backward import BackwardCallback
from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.batch_transform import BatchTransformCallback
from catalyst.callbacks.checkpoint import CheckpointCallback

from catalyst.callbacks.control_flow import ControlFlowCallbackWrapper  # advanced
from catalyst.callbacks.criterion import CriterionCallback
from catalyst.callbacks.metric import (
    BatchMetricCallback,
    FunctionalBatchMetricCallback,
    LoaderMetricCallback,
)
from catalyst.callbacks.metric_aggregation import MetricAggregationCallback
from catalyst.callbacks.misc import (
    CheckRunCallback,
    EarlyStoppingCallback,
    TimerCallback,
    TqdmCallback,
)

from catalyst.callbacks.mixup import MixupCallback  # advanced

# if SETTINGS.onnx_required:
#     from catalyst.callbacks.onnx import OnnxCallback  # advanced, utils

from catalyst.callbacks.optimizer import OptimizerCallback

if SETTINGS.optuna_required:
    from catalyst.callbacks.optuna import OptunaPruningCallback

from catalyst.callbacks.periodic_loader import PeriodicLoaderCallback  # advanced

if LooseVersion(torch.__version__) >= LooseVersion("1.8.1"):
    from catalyst.callbacks.profiler import ProfilerCallback

# if SETTINGS.pruning_required:
#     from catalyst.callbacks.pruning import PruningCallback  # advanced, utils

# if SETTINGS.quantization_required:
#     from catalyst.callbacks.quantization import QuantizationCallback  # advanced, utils


from catalyst.callbacks.scheduler import (
    ILRUpdater,
    LRFinder,
    SchedulerCallback,
)

if SETTINGS.ml_required:
    from catalyst.callbacks.sklearn_model import SklearnModelCallback  # advanced

from catalyst.callbacks.soft_update import SoftUpdateCallaback  # advanced, utils

# from catalyst.callbacks.tracing import TracingCallback  # advanced, utils

from catalyst.callbacks.metrics import *
