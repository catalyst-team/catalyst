Catalyst
======================================

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


PyTorch framework for DL research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop.

Break the cycle - use the Catalyst_!

.. _Catalyst: https://github.com/catalyst-team/catalyst



Installation
------------
Common installation:

.. code:: bash

   pip install -U catalyst


More specific with additional requirements:

.. code:: bash

    pip install catalyst[ml]         # installs DL+ML based catalyst
    pip install catalyst[cv]         # installs DL+CV based catalyst
    pip install catalyst[nlp]        # installs DL+NLP based catalyst
    pip install catalyst[ecosystem]  # installs Catalyst.Ecosystem for DL R&D
    pip install catalyst[contrib]    # installs DL+contrib based catalyst
    pip install catalyst[all]        # installs everything. Very convenient to deploy on a new server


Catalyst is compatible with: Python 3.6+. PyTorch 1.0.0+.


Getting started
------------------------------------------------------

.. code:: python

    import torch
    from catalyst.dl import SupervisedRunner

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
        verbose=True,
    )

For Catalyst.RL introduction, please follow `Catalyst.RL repo`_.

.. _Catalyst.RL repo: https://github.com/catalyst-team/catalyst-rl


Docs and examples
------------------------
1. Detailed `classification tutorial`_
#. Advanced `segmentation tutorial`_
#. Comprehensive `classification pipeline`_
#. Binary and semantic `segmentation pipeline`_

.. _`classification tutorial`: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
.. _`segmentation tutorial`: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb
.. _`classification pipeline`: https://github.com/catalyst-team/classification
.. _`segmentation pipeline`: https://github.com/catalyst-team/segmentation

In the examples_ of the repository, you can find advanced tutorials and Catalyst best practices.

.. _examples: https://github.com/catalyst-team/catalyst/tree/master/examples


Infos
~~~~~~
To learn more about Catalyst internals and to be aware of the most important features, you can read `Catalyst-info`_, our blog where we regularly write facts about the framework.

.. _`Catalyst-info`: https://github.com/catalyst-team/catalyst-info


We also supervise the `Awesome Catalyst list`_ – Catalyst-powered projects, tutorials and talks.
Feel free to make a PR with your project to the list. And don't forget to check out current list, there are many interesting projects.

.. _`Awesome Catalyst list`: https://github.com/catalyst-team/awesome-catalyst-list


Releases
~~~~~~~~~~~~
We deploy a major release once a month with a name like ``YY.MM``.
And micro-releases with framework improvements during a month in the format ``YY.MM.#``.

You can view the changelog on the `GitHub Releases`_ page.

.. _`GitHub Releases`: https://github.com/catalyst-team/catalyst/releases


Overview
--------

Catalyst helps you write compact
but full-featured DL pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.

Features
~~~~~~~~~~~~~~~~~

- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – all source code and environment variables will be saved.
- Callbacks – reusable train/inference pipeline parts.
- Training stages support.
- Easy customization.
- PyTorch best practices (SWA, AdamW, Ranger optimizer, OneCycle, and more).
- Developments best practices - fp16 support, distributed training, slurm

Structure
~~~~~~~~~~~~~~~~~

- **core** - framework core with main abstractions - Experiment, Runner, State, Callback.
- **DL** – runner for training and inference,
   all of the classic ML and CV/NLP metrics
   and a variety of callbacks for training, validation
   and inference of neural networks.
- **contrib** - additional modules contributed by Catalyst users.
- **data** - useful tools and scripts for data processing.


Docker
~~~~~~~~~~~~~~~~~

Catalyst has its own `DockerHub page`_:

.. _`DockerHub page`: https://hub.docker.com/r/catalystteam/catalyst/tags

- ``catalystteam/catalyst:{CATALYST_VERSION}`` – simple image with Catalyst
- ``catalystteam/catalyst:{CATALYST_VERSION}-fp16`` – Catalyst with FP16
- ``catalystteam/catalyst:{CATALYST_VERSION}-dev`` – Catalyst for development with all the requirements
- ``catalystteam/catalyst:{CATALYST_VERSION}-dev-fp16`` – Catalyst for development with FP16


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
        title = {Accelerated DL R&D},
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


.. toctree::
    :caption: Overview
    :maxdepth: 2
    :hidden:

    self
    info/examples
    info/contributing

.. toctree::
    :caption: API

    api/core
    api/dl

    api/contrib
    api/data
    api/utils