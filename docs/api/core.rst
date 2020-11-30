Core
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


.. automodule:: catalyst.core
    :members:
    :show-inheritance:


Experiment
----------------------
.. autoclass:: catalyst.core.experiment.IExperiment
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.core.experiment
    :members:
    :undoc-members:
    :show-inheritance:


Runner
----------------------
.. autoclass:: catalyst.core.runner.RunnerException
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.runner.IRunner
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.runner.IStageBasedRunner
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.core.runner
    :members:
    :undoc-members:
    :show-inheritance:


RunnerLegacy
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.core.legacy.IRunnerLegacy
    :members:
    :undoc-members:
    :show-inheritance:


Callback
----------------------
.. autoclass:: catalyst.core.callback.CallbackNode
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.callback.CallbackOrder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.callback.CallbackScope
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.callback.Callback
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: catalyst.core.callback.CallbackWrapper
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.core.callback
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