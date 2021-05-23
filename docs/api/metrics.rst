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
    :exclude-members: __init__
    :show-inheritance:

ICallbackBatchMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.ICallbackBatchMetric
    :members:
    :exclude-members: __init__
    :show-inheritance:

ICallbackLoaderMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.ICallbackLoaderMetric
    :members:
    :exclude-members: __init__
    :show-inheritance:

AccumulationMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._metric.AccumulationMetric
    :members:
    :exclude-members: __init__
    :show-inheritance:

General Metrics
----------------------

AdditiveValueMetric
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._additive.AdditiveValueMetric
    :members:
    :exclude-members: __init__
    :show-inheritance:

ConfusionMatrixMetric
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._confusion_matrix.ConfusionMatrixMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

FunctionalBatchMetric
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._functional_metric.FunctionalBatchMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:


Runner Metrics
----------------------

Accuracy - AccuracyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._accuracy.AccuracyMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Accuracy - MultilabelAccuracyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._accuracy.MultilabelAccuracyMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

AUCMetric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._auc.AUCMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Classification – BinaryPrecisionRecallF1Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.BinaryPrecisionRecallF1Metric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Classification – MulticlassPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.MulticlassPrecisionRecallF1SupportMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Classification – MultilabelPrecisionRecallF1SupportMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._classification.MultilabelPrecisionRecallF1SupportMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

CMCMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._cmc_score.CMCMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

ReidCMCMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._cmc_score.ReidCMCMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

RecSys – HitrateMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._hitrate.HitrateMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

RecSys – MAPMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._map.MAPMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

RecSys – MRRMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._mrr.MRRMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

RecSys – NDCGMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._ndcg.NDCGMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Segmentation – RegionBasedMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.RegionBasedMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Segmentation – DiceMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.DiceMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Segmentation – IOUMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.IOUMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Segmentation – TrevskyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.metrics._segmentation.TrevskyMetric
    :exclude-members: __init__
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
.. automodule:: catalyst.metrics.functional._recall
    :members:
    :undoc-members:
    :show-inheritance:

Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._segmentation
    :members:
    :undoc-members:
    :show-inheritance:

Misc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.metrics.functional._misc
    :members:
    :undoc-members:
    :show-inheritance: