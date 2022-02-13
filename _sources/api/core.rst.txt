Core
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


.. automodule:: catalyst.core
    :members:
    :show-inheritance:


Runner
----------------------

IRunner
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.runner.IRunner
    :members: seed, hparams, num_epochs, get_engine, get_loggers, get_loaders, get_model, get_criterion, get_optimizer, get_scheduler, get_callbacks, log_metrics, log_image, log_hparams, handle_batch, run
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

IRunnerError
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.runner.IRunnerError
    :members:
    :undoc-members:
    :show-inheritance:

Engine
----------------------
.. autoclass:: catalyst.core.engine.Engine
    :members:
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

Callback
----------------------

ICallback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.ICallback
    :members:
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

CallbackOrder
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackOrder
    :members:
    :undoc-members:
    :show-inheritance:

Callback
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.Callback
    :members:
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

CallbackWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackWrapper
    :members:
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

ILogger
----------------------
.. autoclass:: catalyst.core.logger.ILogger
    :members:
    :undoc-members:
    :show-inheritance:
