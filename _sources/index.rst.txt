Catalyst
========================================

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


PyTorch framework for Deep Learning R&D.
--------------------------------------------------------------------------------

It focuses on reproducibility, rapid experimentation, and codebase reuse
so you can **create** something new rather than write yet another train loop.
Break the cycle - use the Catalyst_!

- `Project Manifest`_
- `Framework architecture`_
- `Catalyst at AI Landscape`_
- Part of the `PyTorch Ecosystem`_

.. _PyTorch Ecosystem: https://pytorch.org/ecosystem/
.. _Catalyst Ecosystem: https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing
.. _Catalyst: https://github.com/catalyst-team/catalyst
.. _`Project Manifest`: https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md
.. _`Framework architecture`: https://miro.com/app/board/o9J_lxBO-2k=/
.. _Catalyst at AI Landscape: https://landscape.lfai.foundation/selected=catalyst

Getting started
----------------------------------------

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl, utils
    from catalyst.contrib.datasets import MNIST

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False), batch_size=32),
    }

    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3, 5)),
            dl.PrecisionRecallF1SupportCallback(input_key="logits", target_key="targets"),
        ],
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )

    # model evaluation
    metrics = runner.evaluate_loader(
        loader=loaders["valid"],
        callbacks=[dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3, 5))],
    )

    # model inference
    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10

    # model post-processing
    model = runner.model.cpu()
    batch = next(iter(loaders["valid"]))[0]
    utils.trace_model(model=model, batch=batch)
    utils.quantize_model(model=model)
    utils.prune_model(model=model, pruning_fn="l1_unstructured", amount=0.8)
    utils.onnx_export(model=model, batch=batch, file="./logs/mnist.onnx", verbose=True)


Step by step guide
~~~~~~~~~~~~~~~~~~~~~~
1. Start with `Catalyst — A PyTorch Framework for Accelerated Deep Learning R&D`_ introduction.
2. Try `notebook tutorials`_ or check `minimal examples`_ for first deep dive.
3. Read `blogposts`_ with use-cases and guides.
4. Learn machine learning with our `"Deep Learning with Catalyst" course`_.
5. And do not forget to `join our slack`_ for collaboration.

.. _`Catalyst — A PyTorch Framework for Accelerated Deep Learning R&D`: https://medium.com/pytorch/catalyst-a-pytorch-framework-for-accelerated-deep-learning-r-d-ad9621e4ca88?source=friends_link&sk=885b4409aecab505db0a63b06f19dcef
.. _`Kittylyst`: https://github.com/Scitator/kittylyst
.. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`notebook tutorials`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`blogposts`: https://catalyst-team.com/post/
.. _`"Deep Learning with Catalyst" course`: https://github.com/catalyst-team/dl-course
.. _`classification`: https://github.com/catalyst-team/classification
.. _`detection`: https://github.com/catalyst-team/detection
.. _`segmentation`: https://github.com/catalyst-team/segmentation
.. _`projects section`: https://github.com/catalyst-team/catalyst#projects
.. _`Alchemy`: https://github.com/catalyst-team/alchemy
.. _`Reaction`: https://github.com/catalyst-team/reaction
.. _`Catalyst.RL repo`: https://github.com/catalyst-team/catalyst-rl
.. _`contribution guidelines`: https://github.com/catalyst-team/catalyst/blob/master/CONTRIBUTING.md
.. _`patreon page`: https://patreon.com/catalyst_team
.. _`write us`: https://github.com/catalyst-team/catalyst#user-feedback
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw

Overview
----------------------------------------
Catalyst helps you write compact
but full-featured Deep Learning pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.


Installation
~~~~~~~~~~~~~~~~~~~~~~
Common installation:

.. code:: bash

   pip install -U catalyst


More specific with additional requirements:

.. code:: bash

    pip install catalyst[ml]         # installs ML-based Catalyst
    pip install catalyst[cv]         # installs CV-based Catalyst
    # master version installation
    pip install git+https://github.com/catalyst-team/catalyst@master --upgrade


Catalyst is compatible with: Python 3.7+. PyTorch 1.4+.

Tested on Ubuntu 16.04/18.04/20.04, macOS 10.15, Windows 10 and Windows Subsystem for Linux.

Tests
~~~~~~~~~~~~~~~~~~~~~~
All Catalyst code, features and pipelines `are fully tested`_
with our own `catalyst-codestyle`_.
During testing, we train a variety of different models: image classification,
image segmentation, text classification, GANs, and much more.
We then compare their convergence metrics in order to verify
the correctness of the training procedure and its reproducibility.
As a result, Catalyst provides fully tested and reproducible
best practices for your deep learning research and development.

.. _are fully tested: https://github.com/catalyst-team/catalyst/tree/master/tests
.. _catalyst-codestyle: https://github.com/catalyst-team/codestyle


Indices and tables
----------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
    :caption: Getting started
    :maxdepth: 2
    :hidden:

    self
    getting_started/quickstart
    Minimal examples <https://github.com/catalyst-team/catalyst#minimal-examples>
    Catalyst — Accelerated Deep Learning R&D <https://medium.com/pytorch/catalyst-a-pytorch-framework-for-accelerated-deep-learning-r-d-ad9621e4ca88?source=friends_link&sk=885b4409aecab505db0a63b06f19dcef>


.. toctree::
    :caption: Tutorials
    :maxdepth: 2
    :hidden:

    tutorials/ddp

.. toctree::
    :caption: Core
    :maxdepth: 2
    :hidden:

    core/callback
    core/engine
    core/logger
    core/metric
    core/runner

.. toctree::
    :caption: FAQ
    :maxdepth: 2
    :hidden:

    faq/intro

    faq/architecture
    faq/checkpointing
    faq/dataflow
    faq/dp
    faq/debugging
    faq/ddp
    faq/early_stopping
    faq/engines
    faq/inference
    faq/logging
    faq/mixed_precision
    faq/multi_components
    faq/multi_keys
    faq/optuna
    faq/settings


.. toctree::
    :caption: API

    api/callbacks
    api/contrib
    api/core
    api/data
    api/engines
    api/loggers
    api/metrics
    api/runners
    api/utils


.. toctree::
    :caption: Contribution guide
    :maxdepth: 2
    :hidden:

    How to start <https://github.com/catalyst-team/catalyst/blob/master/CONTRIBUTING.md>
    Codestyle <https://github.com/catalyst-team/codestyle>
    Acknowledgments <https://github.com/catalyst-team/catalyst#acknowledgments>
