Architecture
==============================================================================

Catalyst framework architecture have the following structure:

.. code:: bash

    catalyst/
        callbacks/
        contrib/
        core/
        data/
        dl/
        engines/
        extras/
        loggers/
        metrics/
        runners/
        utils/
        registy.py
        settings.py

Long story short,

- ``callbacks`` - variety of different for-loop extensions as a mixup, tracing, soft update, etc.
- ``contrib`` - deep learning and reinforcement learning models, losses, layers, etc. Use at your own risk, developing under "code-as-a-documentation" vision.
- ``core`` - core idea of the framework and for-loop wrapper.
- ``data`` - API for useful PyTorch dataset wrappers, samplers and more.
- ``dl`` - main entrypoint, which gives an access to `callbacks`, `core`, `engines`, `loggers`, `runners` simultaneously.
- ``engines`` - Catalyst way to handle different hardware available.
- ``extras`` - Python extras.
- ``loggers`` - API for different moniting systems available (Tensorboard, MLflow, etc).
- ``metrics`` - variety of deep learning metric implementations for classification, regression, segmentaiton, ranking and more.
- ``runners`` - Catalyst primitives for different tasks such as supervised learning, self-supervised learning, reinforcement learning, etc.
- ``utils`` - many PyTorch and Python useful functions for deep learning R&D.
- ``registry.py`` - Catalyst Config API and Registry for yaml-based pipeline creation.
- ``settings.py`` - framework main extension settings.


Entrypoints
----------------------------------------

There are 5 entrypoints to the framework, which are preferale to use:

.. code-block:: python

    from catalyst.contrib import data as cdata, datasets, layers, losses, models, optimizers, schedulers, utils as cutils
    from catalyst import data
    from catalyst import dl
    from catalyst import metrics
    from catalyst import utils

The framework was designed to give you easy access to all the functionality through these entrypoints.
A few examples:

.. code-block:: python

    from catalyst import dl

    runner = dl.SupervisedRunner()

or

.. code-block:: python

    from catalyst import utils

    utils.set_global_seed(42)

There are also 2 extra entrypoints, such as

.. code-block:: python

    from catalyst.settings import SETTIGNS
    from catalyst.registry import Registry

for advanced Catalyst usage.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
