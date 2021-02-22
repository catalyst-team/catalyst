Runners
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


Runner API
----------------------

ISupervisedRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.supervised.ISupervisedRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:


Python API
----------------------

Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.runner.Runner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:


SupervisedRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.supervised.SupervisedRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:


Config API
----------------------

ConfigRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.config.ConfigRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

SupervisedConfigRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.config.SupervisedConfigRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:


Hydra API
----------------------

HydraRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.hydra.HydraRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

SupervisedHydraRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.runners.hydra.SupervisedHydraRunner
    :members:
    :exclude-members: on_experiment_start, on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_stage_end, on_experiment_end
    :undoc-members:
    :show-inheritance:

