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
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CheckpointCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.checkpoint.CheckpointCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ControlFlowCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.control_flow.ControlFlowCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Criterion
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.criterion.CriterionCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – BatchMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.BatchMetricCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – LoaderMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.LoaderMetricCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric – MetricAggregationCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric_aggregation.MetricAggregationCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – CheckRunCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.CheckRunCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – EarlyStoppingCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.EarlyStoppingCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – TimerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TimerCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Misc – TqdmCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TqdmCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

OptimizerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.optimizer.OptimizerCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

OptunaPruningCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.optuna.OptunaPruningCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

PeriodicLoaderCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.periodic_loader.PeriodicLoaderCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

PruningCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.pruning.PruningCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

.. Quantization

Scheduler – SchedulerCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.SchedulerCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Scheduler – LRFinder
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.LRFinder
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

.. Tracing

BatchTransformCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.transform.BatchTransformCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:


Metric
----------------------
.. automodule:: catalyst.callbacks.metrics
    :members:
    :show-inheritance:


AccuracyCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.accuracy.AccuracyCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

AUCCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.auc.AUCCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

.. Classification

.. CMC score

ConfusionMatrixCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.confusion_matrix.ConfusionMatrixCallback
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

.. Segmentation
