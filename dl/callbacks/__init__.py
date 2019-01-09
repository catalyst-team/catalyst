from typing import List, Union

from catalyst.contrib import Factory, register

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
    "LRUpdater", "OneCycleLR", "LRFinder", "register_callback", "CALLBACKS"
]

CALLBACKS = {
    **core.__dict__,
    **metrics.__dict__,
    **schedulers.__dict__,
}


def register_callback(
    *callback_factories: Factory
) -> Union[Factory, List[Factory]]:
    """Add callback type or factory method to global
        callback list to make it available in config
        Can be called or used as decorator
        :param: callback_factories Required criterion factory (method or type)
        :returns: single callback factory or list of them
    """
    return register("callbacks")(*callback_factories)
