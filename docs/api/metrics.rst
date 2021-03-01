Metrics
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


.. automodule:: catalyst.metrics
    :members:
    :show-inheritance:


Metric API
----------------------

IMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.metric.IMetric
    :members:
    :undoc-members:
    :show-inheritance:

ICallbackBatchMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.metric.ICallbackBatchMetric
    :members:
    :undoc-members:
    :show-inheritance:

ICallbackLoaderMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.metric.ICallbackLoaderMetric
    :members:
    :undoc-members:
    :show-inheritance:

General Metrics
----------------------

AdditiveValueMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.additive.AdditiveValueMetric
    :members:
    :undoc-members:
    :show-inheritance:

ConfusionMatrixMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.confusion_matrix.ConfusionMatrixMetric
    :members:
    :undoc-members:
    :show-inheritance:


Runner Metrics
----------------------

AccuracyMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.accuracy.AccuracyMetric
    :members:
    :undoc-members:
    :show-inheritance:

MultilabelAccuracyMetric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.accuracy.MultilabelAccuracyMetric
    :members:
    :undoc-members:
    :show-inheritance:

AUCMetric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.auc.AUCMetric
    :members:
    :undoc-members:
    :show-inheritance:

BinaryPrecisionRecallF1Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.classification.BinaryPrecisionRecallF1Metric
    :members:
    :undoc-members:
    :show-inheritance:

MulticlassPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.classification.MulticlassPrecisionRecallF1SupportMetric
    :members:
    :undoc-members:
    :show-inheritance:

MultilabelPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.classification.MultilabelPrecisionRecallF1SupportMetric
    :members:
    :undoc-members:
    :show-inheritance:

HitrateMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.hitrate.HitrateMetric
    :members:
    :undoc-members:
    :show-inheritance:

NDCGMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.ndcg.NDCGMetric
    :members:
    :undoc-members:
    :show-inheritance:

MAPMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.map.MAPMetric
    :members:
    :undoc-members:
    :show-inheritance:

MRRMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.mrr.MRRMetric
    :members:
    :undoc-members:
    :show-inheritance:

RegionBasedMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.segmentation.RegionBasedMetric
    :members:
    :undoc-members:
    :show-inheritance:

IOUMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.segmentation.IOUMetric
    :members:
    :undoc-members:
    :show-inheritance:

JaccardMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.segmentation.JaccardMetric
    :members:
    :undoc-members:
    :show-inheritance:

DiceMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.segmentation.DiceMetric
    :members:
    :undoc-members:
    :show-inheritance:

TrevskyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics.segmentation.TrevskyMetric
    :members:
    :undoc-members:
    :show-inheritance:


Functional API
----------------------

Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.accuracy
    :members:
    :undoc-members:
    :show-inheritance:

AUC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.auc
    :members:
    :undoc-members:
    :show-inheritance:

Average Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.average_precision
    :members:
    :undoc-members:
    :show-inheritance:

Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.classification
    :members:
    :undoc-members:
    :show-inheritance:

CMC Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.cmc_score
    :members:
    :undoc-members:
    :show-inheritance:

F1 score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.f1_score
    :members:
    :undoc-members:
    :show-inheritance:

Focal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.focal
    :members:
    :undoc-members:
    :show-inheritance:

Hitrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.hitrate
    :members:
    :undoc-members:
    :show-inheritance:

Misc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.misc
    :members:
    :undoc-members:
    :show-inheritance:

MRR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.mrr
    :members:
    :undoc-members:
    :show-inheritance:

NDCG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.ndcg
    :members:
    :undoc-members:
    :show-inheritance:

Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.precision
    :members:
    :undoc-members:
    :show-inheritance:

Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.recall
    :members:
    :undoc-members:
    :show-inheritance:

Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional.segmentation
    :members:
    :undoc-members:
    :show-inheritance:
