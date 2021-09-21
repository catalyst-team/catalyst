# flake8: noqa
# import order:
# functional
# core metrics
# metrics

from catalyst.metrics.functional import *

from catalyst.metrics._metric import (
    ICallbackBatchMetric,
    ICallbackLoaderMetric,
    IMetric,
)
from catalyst.metrics._accumulative import AccumulativeMetric
from catalyst.metrics._additive import AdditiveMetric, AdditiveValueMetric
from catalyst.metrics._confusion_matrix import ConfusionMatrixMetric
from catalyst.metrics._functional_metric import FunctionalBatchMetric, FunctionalLoaderMetric

from catalyst.metrics._accuracy import AccuracyMetric, MultilabelAccuracyMetric
from catalyst.metrics._auc import AUCMetric
from catalyst.metrics._classification import (
    BinaryPrecisionRecallF1Metric,
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)
from catalyst.metrics._cmc_score import CMCMetric, ReidCMCMetric
from catalyst.metrics._hitrate import HitrateMetric
from catalyst.metrics._map import MAPMetric
from catalyst.metrics._mrr import MRRMetric
from catalyst.metrics._ndcg import NDCGMetric
from catalyst.metrics._r2_squared import R2Squared
from catalyst.metrics._segmentation import (
    RegionBasedMetric,
    IOUMetric,
    DiceMetric,
    TrevskyMetric,
)
