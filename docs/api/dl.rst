DL
===========

.. automodule:: catalyst.dl
    :members:
    :undoc-members:
    :show-inheritance:


Runner
----------

.. automodule:: catalyst.dl.experiments.runner
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.dl.state
    :members:
    :undoc-members:
    :show-inheritance:

Experiment
----------

.. automodule:: catalyst.dl.experiments.experiment
    :members:
    :undoc-members:
    :show-inheritance:


Metrics
----------

.. automodule:: catalyst.dl.metrics
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.dl.metric_manager
    :members:
    :undoc-members:
    :show-inheritance:


Callbacks
----------
Callbacks are the main idea for reproducible pipeline

.. currentmodule:: catalyst.dl.callbacks

Base
~~~~~~~~~~

.. autoclass:: Callback
    :members:
    :undoc-members:

.. autoclass:: LossCallback
    :members:
    :undoc-members:

.. autoclass:: OptimizerCallback
    :members:
    :undoc-members:

.. autoclass:: SchedulerCallback
    :members:
    :undoc-members:

.. autoclass:: CheckpointCallback
    :members:
    :undoc-members:

.. autoclass:: EarlyStoppingCallback
    :members:
    :undoc-members:


Metrics
~~~~~~~~~~

.. autoclass:: MetricCallback
    :members:
    :undoc-members:

.. autoclass:: DiceCallback
    :members:
    :undoc-members:


.. autoclass:: IouCallback
    :members:
    :undoc-members:

.. autoclass:: F1ScoreCallback
    :members:
    :undoc-members:

.. autoclass:: AUCCallback
    :members:
    :undoc-members:

.. autoclass:: ConfusionMatrixCallback
    :members:
    :undoc-members:


MultiMetrics
~~~~~~~~~~~~~~

.. autoclass:: MultiMetricCallback
    :members:
    :undoc-members:

.. autoclass:: AccuracyCallback
    :members:
    :undoc-members:

.. autoclass:: MapKCallback
    :members:
    :undoc-members:


Loggers
~~~~~~~~~~~~~~

.. autoclass:: TensorboardLogger
    :members:
    :undoc-members:

.. autoclass:: ConsoleLogger
    :members:
    :undoc-members:

.. autoclass:: VerboseLogger
    :members:
    :undoc-members:

Formatters
""""""""""""""""""

.. autoclass:: MetricsFormatter
    :members:
    :undoc-members:

.. autoclass:: TxtMetricsFormatter
    :members:
    :undoc-members:

.. autoclass:: JsonMetricsFormatter
    :members:
    :undoc-members:


Schedulers
~~~~~~~~~~~~~~

.. autoclass:: LRUpdater
    :members:
    :undoc-members:

.. autoclass:: LRFinder
    :members:
    :undoc-members:


Inference
~~~~~~~~~~~~~~

.. autoclass:: InferCallback
    :members:
    :undoc-members:

.. autoclass:: InferMaskCallback
    :members:
    :undoc-members:


Utils
~~~~~~~~~~

.. automodule:: catalyst.dl.callbacks.utils
    :members:
    :undoc-members:

Losses
-----------------

.. automodule:: catalyst.dl.losses
    :members:
    :undoc-members:
    :show-inheritance:

Metrics
-----------------
Metric functions

.. automodule:: catalyst.dl.metrics
    :members:
    :undoc-members:
    :show-inheritance:

Initialization
-----------------

.. automodule:: catalyst.dl.initialization
    :members:
    :undoc-members:
    :show-inheritance:

Tracing
----------

.. automodule:: catalyst.dl.utils.trace
    :members:
    :undoc-members:
    :show-inheritance:

Utils
----------

.. automodule:: catalyst.dl.utils
    :members:
    :undoc-members:
    :show-inheritance:
