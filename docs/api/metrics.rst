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
.. autoclass:: catalyst.metrics._metric.IMetric
    :members:
    :undoc-members:
    :show-inheritance:

ICallbackBatchMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.ICallbackBatchMetric
    :members:
    :undoc-members:
    :show-inheritance:

ICallbackLoaderMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.ICallbackLoaderMetric
    :members:
    :undoc-members:
    :show-inheritance:

AccumulationMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.AccumulationMetric
    :members:
    :undoc-members:
    :show-inheritance:

General Metrics
----------------------

AdditiveValueMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._additive.AdditiveValueMetric
    :members:
    :undoc-members:
    :show-inheritance:

ConfusionMatrixMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._confusion_matrix.ConfusionMatrixMetric
    :members:
    :undoc-members:
    :show-inheritance:

BatchFunctionalMetric
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._functional_metric.BatchFunctionalMetric
    :members:
    :undoc-members:
    :show-inheritance:


Runner Metrics
----------------------

AccuracyMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._accuracy.AccuracyMetric
    :members:
    :undoc-members:
    :show-inheritance:

MultilabelAccuracyMetric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._accuracy.MultilabelAccuracyMetric
    :members:
    :undoc-members:
    :show-inheritance:

AUCMetric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._auc.AUCMetric
    :members:
    :undoc-members:
    :show-inheritance:

CMCMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._cmc_score.CMCMetric
    :members:
    :undoc-members:
    :show-inheritance:

HitrateMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._hitrate.HitrateMetric
    :members:
    :undoc-members:
    :show-inheritance:

NDCGMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._ndcg.NDCGMetric
    :members:
    :undoc-members:
    :show-inheritance:

MAPMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._map.MAPMetric
    :members:
    :undoc-members:
    :show-inheritance:

MRRMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._mrr.MRRMetric
    :members:
    :undoc-members:
    :show-inheritance:

BinaryPrecisionRecallF1Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.BinaryPrecisionRecallF1Metric
    :members:
    :undoc-members:
    :show-inheritance:

MulticlassPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.MulticlassPrecisionRecallF1SupportMetric
    :members:
    :undoc-members:
    :show-inheritance:

MultilabelPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.MultilabelPrecisionRecallF1SupportMetric
    :members:
    :undoc-members:
    :show-inheritance:

RegionBasedMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.RegionBasedMetric
    :members:
    :undoc-members:
    :show-inheritance:

IOUMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.IOUMetric
    :members:
    :undoc-members:
    :show-inheritance:

JaccardMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.JaccardMetric
    :members:
    :undoc-members:
    :show-inheritance:

DiceMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.DiceMetric
    :members:
    :undoc-members:
    :show-inheritance:

TrevskyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.TrevskyMetric
    :members:
    :undoc-members:
    :show-inheritance:


Functional API
----------------------

Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._accuracy
    :members:
    :undoc-members:
    :show-inheritance:

AUC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._auc
    :members:
    :undoc-members:
    :show-inheritance:

Average Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._average_precision
    :members:
    :undoc-members:
    :show-inheritance:

Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._classification
    :members:
    :undoc-members:
    :show-inheritance:

CMC Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._cmc_score
    :members:
    :undoc-members:
    :show-inheritance:

F1 score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._f1_score
    :members:
    :undoc-members:
    :show-inheritance:

Focal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._focal
    :members:
    :undoc-members:
    :show-inheritance:

Hitrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._hitrate
    :members:
    :undoc-members:
    :show-inheritance:

Misc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._misc
    :members:
    :undoc-members:
    :show-inheritance:

MRR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._mrr
    :members:
    :undoc-members:
    :show-inheritance:

NDCG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._ndcg
    :members:
    :undoc-members:
    :show-inheritance:

Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._precision
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
.. automodule:: catalyst.metrics.functional._segmentation
    :members:
    :undoc-members:
    :show-inheritance:
