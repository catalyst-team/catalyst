from . import core
from . import metrics
from . import schedulers
from .core import Callback, CallbackCompose, Logger, \
    TensorboardLogger, CheckpointCallback, OptimizerCallback, \
    SchedulerCallback, ClassificationLossCallback, InferCallback, \
    MixupCallback, InferMaskCallback
from .metrics import MetricCallback, MultiMetricCallback, \
    DiceCallback, JaccardCallback, PrecisionCallback, MapKCallback
from .schedulers import LRUpdater, OneCycleLR, LRFinder

__all__ = [
    "Callback", "CallbackCompose", "Logger", "TensorboardLogger",
    "CheckpointCallback", "OptimizerCallback", "SchedulerCallback",
    "ClassificationLossCallback", "InferCallback", "MixupCallback",
    "InferMaskCallback", "MetricCallback", "MultiMetricCallback",
    "DiceCallback", "JaccardCallback", "PrecisionCallback", "MapKCallback",
    "LRUpdater", "OneCycleLR", "LRFinder", "CALLBACKS"
]

CALLBACKS = {
    **core.__dict__,
    **metrics.__dict__,
    **schedulers.__dict__,
}
