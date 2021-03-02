# flake8: noqa

import logging

logger = logging.getLogger(__name__)

from catalyst.settings import SETTINGS

from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.callbacks.metrics.auc import AUCCallback

try:
    import matplotlib  # noqa: F401

    from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
except ModuleNotFoundError as ex:
    if SETTINGS.ml_required:
        logger.warning(
            "catalyst[ml] requirements are not available, to install them,"
            " run `pip install catalyst[ml]`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.ml_required:
        logger.warning(
            "catalyst[ml] requirements are not available, to install them,"
            " run `pip install catalyst[ml]`."
        )
        raise ex


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
