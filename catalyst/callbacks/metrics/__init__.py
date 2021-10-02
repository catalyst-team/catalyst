# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.callbacks.metrics.accuracy import AccuracyCallback, MultilabelAccuracyCallback
from catalyst.callbacks.metrics.auc import AUCCallback

from catalyst.callbacks.metrics.classification import (
    PrecisionRecallF1SupportCallback,
    MultilabelPrecisionRecallF1SupportCallback,
)

from catalyst.callbacks.metrics.cmc_score import CMCScoreCallback, ReidCMCScoreCallback

if SETTINGS.ml_required:
    from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback

from catalyst.callbacks.metrics.functional_metric import FunctionalMetricCallback

from catalyst.callbacks.metrics.r2_squared import R2SquaredCallback

from catalyst.callbacks.metrics.recsys import (
    HitrateCallback,
    MAPCallback,
    MRRCallback,
    NDCGCallback,
)

from catalyst.callbacks.metrics.segmentation import (
    DiceCallback,
    IOUCallback,
    TrevskyCallback,
)

if SETTINGS.ml_required:
    from catalyst.callbacks.metrics.scikit_learn import SklearnBatchCallback, SklearnLoaderCallback
