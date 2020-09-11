# flake8: noqa

from catalyst.tools.settings import (
    IS_QUANTIZATION_AVAILABLE,
    IS_PRUNING_AVAILABLE,
)

from catalyst.core.callbacks import *
from catalyst.dl.callbacks.confusion_matrix import ConfusionMatrixCallback
from catalyst.dl.callbacks.inference import InferCallback
from catalyst.dl.callbacks.meter import MeterMetricsCallback
from catalyst.dl.callbacks.mixup import MixupCallback
from catalyst.dl.callbacks.scheduler import LRFinder
from catalyst.dl.callbacks.tracing import TracerCallback
from catalyst.dl.callbacks.metrics import *

if IS_QUANTIZATION_AVAILABLE:
    from catalyst.dl.callbacks.quantization import DynamicQuantizationCallback

if IS_PRUNING_AVAILABLE:
    from catalyst.dl.callbacks.pruning import PruningCallback

from catalyst.contrib.dl.callbacks import *
