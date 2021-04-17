Callbacks
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


Run
----------------------
.. automodule:: catalyst.callbacks
    :members:
    :show-inheritance:


BatchOverfitCallback
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.batch_overfit.BatchOverfitCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

BatchTransformCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.batch_transform.BatchTransformCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CheckpointCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.checkpoint.CheckpointCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ControlFlowCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.control_flow.ControlFlowCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CriterionCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.criterion.CriterionCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – BatchMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.BatchMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – LoaderMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.LoaderMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – MetricAggregationCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric_aggregation.MetricAggregationCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – CheckRunCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.CheckRunCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – EarlyStoppingCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.EarlyStoppingCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – TimerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TimerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – TqdmCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TqdmCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

OnnxCallback
~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.onnx.OnnxCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

OptimizerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.optimizer.OptimizerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

OptunaPruningCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.optuna.OptunaPruningCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

PeriodicLoaderCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.periodic_loader.PeriodicLoaderCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

PruningCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.pruning.PruningCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

QuantizationCallback
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.quantization.QuantizationCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Scheduler – SchedulerCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.SchedulerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Scheduler – LRFinder
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.LRFinder
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Tracing
~~~~~~~
.. autoclass:: catalyst.callbacks.tracing.TracingCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:


Metric
----------------------
.. automodule:: catalyst.callbacks.metrics
    :members:
    :show-inheritance:


Accuracy - AccuracyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.accuracy.AccuracyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Accuracy - MultilabelAccuracyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.accuracy.MultilabelAccuracyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

AUCCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.auc.AUCCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Classification – PrecisionRecallF1SupportCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.classification.PrecisionRecallF1SupportCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Classification – MultilabelPrecisionRecallF1SupportCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.classification.MultilabelPrecisionRecallF1SupportCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CMCScoreCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.cmc_score.CMCScoreCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ReidCMCScoreCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.cmc_score.ReidCMCScoreCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ConfusionMatrixCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.confusion_matrix.ConfusionMatrixCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

FunctionalMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.functional_metric.FunctionalMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

RecSys – HitrateCallback
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.HitrateCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

RecSys – MAPCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.MAPCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

RecSys – MRRCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.MRRCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

RecSys – NDCGCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.NDCGCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Segmentation – DiceCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.DiceCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Segmentation – IOUCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.IOUCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Segmentation – TrevskyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.TrevskyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:
