Callbacks
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


Run-based
------------------------------
.. automodule:: catalyst.callbacks
    :members:
    :show-inheritance:


BackwardCallback
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.backward.BackwardCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
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

CheckRunCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.CheckRunCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ControlFlowCallbackWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.control_flow.ControlFlowCallbackWrapper
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CriterionCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.criterion.CriterionCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

EarlyStoppingCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.EarlyStoppingCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end, handle_score_is_better, handle_score_is_not_better
    :show-inheritance:

LRFinder
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.LRFinder
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end, calc_lr, calc_momentum
    :show-inheritance:

MetricAggregationCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric_aggregation.MetricAggregationCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

MixupCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.mixup.MixupCallback
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

ProfilerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.profiler.ProfilerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

SchedulerCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.scheduler.SchedulerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end, make_batch_step, make_epoch_step
    :show-inheritance:

TimerCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TimerCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

TqdmCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.misc.TqdmCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric-based Interfaces
------------------------------

BatchMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.BatchMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

LoaderMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metric.LoaderMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

Metric-based
------------------------------
.. automodule:: catalyst.callbacks.metrics
    :members:
    :show-inheritance:


AccuracyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.accuracy.AccuracyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

AUCCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.auc.AUCCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

CMCScoreCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.cmc_score.CMCScoreCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ConfusionMatrixCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.confusion_matrix.ConfusionMatrixCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

DiceCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.DiceCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

FunctionalMetricCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.functional_metric.FunctionalMetricCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

HitrateCallback
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.HitrateCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

IOUCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.IOUCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

MAPCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.MAPCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

MultilabelAccuracyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.accuracy.MultilabelAccuracyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

MultilabelPrecisionRecallF1SupportCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.classification.MultilabelPrecisionRecallF1SupportCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

MRRCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.MRRCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

NDCGCallback
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.recsys.NDCGCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

PrecisionRecallF1SupportCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.classification.PrecisionRecallF1SupportCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

R2SquaredCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.r2_squared.R2SquaredCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

ReidCMCScoreCallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.cmc_score.ReidCMCScoreCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

SklearnBatchCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.scikit_learn.SklearnBatchCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

SklearnLoaderCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.scikit_learn.SklearnLoaderCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

SklearnModelCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.sklearn_model.SklearnModelCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:

TrevskyCallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.callbacks.metrics.segmentation.TrevskyCallback
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :show-inheritance:
