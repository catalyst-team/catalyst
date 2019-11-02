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

.. image:: https://ucde0995747f356870f615ffb990.previews.dropboxusercontent.com/p/thumb/AAju2yA3zKEEfV1Rbe1hdCK94o5cVH5blrqQCBfy1BFudg8VfehnZrvBCpKEKUjZ0yce8rVWsXDlxCV2tmXL1f18h9VMod21hbQ-E7_X_Qbomca3PLeTe0pTgcfqs1gGef9JBs4y36-raLf2Qrkf_AJGdvUWscUd9OScOHYI8FyrjmF6pqVaMRnJGv8hmfg1QiT1ZjF2I1KqFMiDNxY3CvVltWNYnCltOk0mLG95yUBNlzJIOROCujlKRV1nAsoL6u7f_ynoVJBVmLsnTZeJ4izf10zCdGc5vmxxMRBTxxwZV4OPDuA7jlTfxB2983Ho5h0CzRGa3k6HwWsLmVUfU2Prno8-6UT99q2x3Lq2RXWaT8CbJe7FNg1LbI1WQWq-6_9oQA4JAOXjP_mbWXk721kz/p.png
    :target: https://www.patreon.com/catalyst_team
    :alt: Donate

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst\_logo.png
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

Catalyst is compatible with: Python 3.6+. PyTorch 0.4.1+.

API documentation and an overview of the library can be found here.


Examples
--------
Examples and tutorials could be found :doc:`here </info/examples>`.
In the examples folder of the repository,
you can find advanced tutorials and Catalyst best practices.


Installation
------------

.. code:: bash

   pip install catalyst

Overview
--------

Catalyst helps you write compact
but full-featured DL & RL pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.

Features
^^^^^^^^

-  Universal train/inference loop.
-  Configuration files for model/data hyperparameters.
-  Reproducibility – even source code will be saved.
-  Callbacks – reusable train/inference pipeline parts.
-  Training stages support.
-  Easy customization.
-  PyTorch best practices (SWA, AdamW, 1Cycle, FP16 and more).

Structure
^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Please see the :doc:`contribution guide </info/contributing>`
for more information.


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
