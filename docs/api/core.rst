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
.. autoclass:: catalyst.core.runner.IRunner
    :members: seed, hparams, stages, get_stage_len, get_trial, get_engine, get_loggers, get_datasets, get_loaders, get_model, get_criterion, get_optimizer, get_scheduler, get_callbacks, log_metrics, log_image, log_hparams, handle_batch, run
    :exclude-members: __init__, on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.runner.RunnerException
    :members:
    :undoc-members:
    :show-inheritance:

Engine
----------------------
.. autoclass:: catalyst.core.engine.IEngine
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

CallbackNode
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackNode
    :members:
    :undoc-members:
    :show-inheritance:

CallbackOrder
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackOrder
    :members:
    :undoc-members:
    :show-inheritance:

CallbackScope
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackScope
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

CallbackList
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.callback.CallbackList
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

Logger
----------------------
.. autoclass:: catalyst.core.logger.ILogger
    :members:
    :undoc-members:
    :show-inheritance:

Trial
----------------------
.. autoclass:: catalyst.core.trial.ITrial
    :members:
    :undoc-members:
    :show-inheritance:

Scripts
--------------------------------------

You can use Catalyst scripts with `catalyst-dl` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-dl run --help

.. automodule:: catalyst.dl.__main__
    :members:
    :exclude-members: build_parser, main