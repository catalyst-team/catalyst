Catalyst
======================================

.. image:: https://travis-ci.com/catalyst-team/catalyst.svg?branch=master
    :target: https://travis-ci.com/catalyst-team/catalyst
    :alt: Build Status

.. image:: https://img.shields.io/github/license/catalyst-team/catalyst.svg
    :alt: License

.. image:: https://img.shields.io/pypi/v/catalyst.svg
    :target: https://pypi.org/project/catalyst/
    :alt: Pipi version

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://catalyst-team.github.io/catalyst/index.html
    :alt: Docs

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png
    :target: https://www.patreon.com/catalyst_team
    :alt: Donate

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


High-level utils for PyTorch DL & RL research.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather then write another regular train loop.

Break the cycle - use the Catalyst_!

.. _Catalyst: https://github.com/catalyst-team/catalyst

------------


.. toctree::
   :caption: Overview
   :maxdepth: 1

   self
   info/examples
   info/contributing
   info/license

Catalyst is compatible with: Python 3.6+. PyTorch 1.0.0+.


Installation
------------
Common installation:

.. code:: bash

   pip install -U catalyst


More specific with additional requirements:

.. code:: bash

    pip install catalyst[dl] # installs DL based catalyst with Weights & Biases support
    pip install catalyst[rl] # installs DL+RL based catalyst
    pip install catalyst[drl] # installs DL+RL based catalyst with Weights & Biases support
    pip install catalyst[contrib] # installs DL+contrib based catalyst
    pip install catalyst[all] # installs everything. Very convenient to deploy on a new server


Catalyst is compatible with: Python 3.6+. PyTorch 1.0.0+.

Docs and examples
------------------------
Detailed classification tutorial

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
    :alt: Open In Colab

Advanced segmentation tutorial

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb
    :alt: Open In Colab


Comprehensive `classification pipeline`_

.. _`classification pipeline`: https://github.com/catalyst-team/classification


Binary and semantic `segmentation pipeline`_

.. _`segmentation pipeline`: https://github.com/catalyst-team/segmentation


In the examples_ of the repository, you can find advanced tutorials and Catalyst best practices.

.. _examples: https://github.com/catalyst-team/catalyst/tree/master/examples


Blog
~~~~~~
To learn more about Catalyst internals and to be aware of the most important features, you can read `Catalyst-info`_, our blog where we regularly write facts about the framework.

.. _`Catalyst-info`: https://github.com/catalyst-team/catalyst-info

Awesome list of Catalyst-powered repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We supervise the `Awesome Catalyst list`_. You can make a PR with your project to the list.

.. _`Awesome Catalyst list`: https://github.com/catalyst-team/awesome-catalyst-list

Releases
~~~~~~~~~~~~
We release a major release once a month with a name like YY.MM.
And micro-releases with hotfixes and framework improvements in the format YY.MM.#.

You can view the changelog on the `GitHub Releases`_ page.

.. _`GitHub Releases`: https://github.com/catalyst-team/catalyst/releases

Overview
--------

Catalyst helps you write compact
but full-featured DL & RL pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.

Features
~~~~~~~~~~~~~~~~~

-  Universal train/inference loop.
-  Configuration files for model/data hyperparameters.
-  Reproducibility – all source code and environment variables will be saved.
-  Callbacks – reusable train/inference pipeline parts.
-  Training stages support.
-  Easy customization.
- PyTorch best practices (SWA, AdamW, Ranger optimizer, OneCycleLRWithWarmup, FP16 and more)

Structure
~~~~~~~~~~~~~~~~~

-  **DL** – runner for training and inference,
   all of the classic machine learning and computer vision metrics
   and a variety of callbacks for training, validation
   and inference of neural networks.
-  **RL** – scalable Reinforcement Learning,
   on-policy & off-policy algorithms and their improvements
   with distributed training support.
-  **contrib** - additional modules contributed by Catalyst users.
-  **data** - useful tools and scripts for data processing.


Getting started: 30 seconds with Catalyst
------------------------------------------------------

.. code:: python

    import torch
    from catalyst.dl.experiments import SupervisedRunner

    # experiment setup
    logdir = "./logdir"
    num_epochs = 42

    # data
    loaders = {"train": ..., "valid": ...}

    # model, criterion, optimizer
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )


.. toctree::
   :maxdepth: 2
   :caption: API

   api/dl
   api/rl

   api/contrib
   api/data
   api/utils


Contribution guide
------------------

We appreciate all contributions.
If you are planning to contribute back bug-fixes,
please do so without any further discussion.
If you plan to contribute new features, utility functions or extensions,
please first open an issue and discuss the feature with us.

Please see the `contribution guide`_ for more information.

.. _`contribution guide`: https://github.com/catalyst-team/catalyst/blob/master/CONTRIBUTING.md

By participating in this project, you agree to abide by its `Code of Conduct`_.

.. _`Code of Conduct`: https://github.com/catalyst-team/catalyst/blob/master/CODE_OF_CONDUCT.md

.. image:: https://c5.patreon.com/external/logo/become_a_patron_button.png
    :target: https://www.patreon.com/catalyst_team
    :alt: Donate


License
------------------

This project is licensed under the Apache License, Version 2.0 see the LICENSE_ file for details

.. _LICENSE: https://github.com/catalyst-team/catalyst/blob/master/LICENSE

Citation
------------------

Please use this bibtex if you want to cite this repository in your publications:

.. code:: bibtex

    @misc{catalyst,
        author = {Kolesnikov, Sergey},
        title = {Reproducible and fast DL & RL.},
        year = {2018},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/catalyst-team/catalyst}},
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
