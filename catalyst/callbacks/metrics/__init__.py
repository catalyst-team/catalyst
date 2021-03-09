# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.callbacks.metrics.accuracy import AccuracyCallback, MultilabelAccuracyCallback
from catalyst.callbacks.metrics.auc import AUCCallback
from catalyst.callbacks.metrics.custom import CustomMetricCallback

from catalyst.callbacks.metrics.classification import (
    PrecisionRecallF1SupportCallback,
    MultilabelPrecisionRecallF1SupportCallback,
)

from catalyst.callbacks.metrics.cmc_score import CMCScoreCallback

if SETTINGS.ml_required:
    from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback

from catalyst.callbacks.metrics.recsys import (
    HitrateCallback,
    MAPCallback,
    MRRCallback,
    NDCGCallback,
)

from catalyst.callbacks.metrics.segmentation import (
    IOUCallback,
    JaccardCallback,
    DiceCallback,
    TrevskyCallback,
)

#
# from catalyst.callbacks.metrics.accuracy import (
#     AccuracyCallback,
#     MultiLabelAccuracyCallback,
# )
# from catalyst.callbacks.metrics.auc import AUCCallback
# from catalyst.callbacks.metrics.cmc_score import CMCScoreCallback
#
# from catalyst.callbacks.metrics.dice import (
#     DiceCallback,
#     MulticlassDiceMetricCallback,
# )
#
# from catalyst.callbacks.metrics.f1_score import F1ScoreCallback
# from catalyst.callbacks.metrics.iou import (
#     IouCallback,
#     JaccardCallback,
# )
# from catalyst.callbacks.metrics.mrr import MRRCallback
# from catalyst.callbacks.metrics.perplexity import PerplexityCallback
# from catalyst.callbacks.metrics.ppv_tpr_f1 import PrecisionRecallF1ScoreCallback
# from catalyst.callbacks.metrics.precision import (
#     AveragePrecisionCallback,
#     PrecisionCallback,
# )
# from catalyst.callbacks.metrics.recall import RecallCallback
