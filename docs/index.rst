Catalyst
========================================

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


PyTorch framework for Deep Learning R&D.
--------------------------------------------------------------------------------

It focuses on reproducibility, rapid experimentation, and codebase reuse
so you can **create** something new rather than write another regular train loop.
Break the cycle - use the Catalyst_!

Project manifest_. Part of `PyTorch Ecosystem`_. Part of `Catalyst Ecosystem`_:
    - Alchemy_ - experiments logging & visualization
    - Catalyst_ - accelerated deep learning R&D
    - Reaction_ - convenient deep learning models serving

`Catalyst at AI Landscape`_.

.. _PyTorch Ecosystem: https://pytorch.org/ecosystem/
.. _Catalyst Ecosystem: https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing
.. _Alchemy: https://github.com/catalyst-team/alchemy
.. _Catalyst: https://github.com/catalyst-team/catalyst
.. _Reaction: https://github.com/catalyst-team/reaction
.. _manifest: https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md
.. _Catalyst at AI Landscape: https://landscape.lfai.foundation/selected=catalyst

Getting started
----------------------------------------

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from catalyst import dl, metrics

    model = torch.nn.Linear(28 * 28, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def _handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            y_hat = self.model(x.view(x.size(0), -1))

            loss = F.cross_entropy(y_hat, y)
            accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
            self.batch_metrics.update(
                {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
            )

            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    runner = CustomRunner()
    # model training
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=5,
        verbose=True,
        load_best_on_end=True,
    )
    # model inference
    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10
    # model tracing
    traced_model = runner.trace(loader=loaders["valid"])


Step by step guide
~~~~~~~~~~~~~~~~~~~~~~
1. Start with `Catalyst 101 — Accelerated PyTorch`_ introduction.
2. Check `minimal examples`_.
3. Try `notebook tutorials with Google Colab`_.
4. Read `blogposts`_ with use-cases and guides.
5. Learn machine learning with our `"Deep Learning with Catalyst" course`_.
6. If you would like to contribute to the project, follow our `contribution guidelines`_.
7. If you want to support the project, feel free to donate on `patreon page`_ or `write us`_ with your proposals.
8. And do not forget to `join our slack`_ for collaboration.

.. _`Catalyst 101 — Accelerated PyTorch`: https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92?source=friends_link&sk=d3dd9b2b23500eca046361187b4619ff
.. _`Kittylyst`: https://github.com/Scitator/kittylyst
.. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`Notebook Tutorials with Google Colab`: https://github.com/catalyst-team/catalyst#tutorials
.. _`blogposts`: https://github.com/catalyst-team/catalyst#blogposts
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


Catalyst is compatible with: Python 3.6+. PyTorch 1.3+.

Tested on Ubuntu 16.04/18.04/20.04, macOS 10.15, Windows 10 and Windows Subsystem for Linux.


Features
~~~~~~~~~~~~~~~~~~~~~~
- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – all source code and environment variables will be saved.
- Callbacks – reusable train/inference pipeline parts with easy customization.
- Training stages support.
- Deep Learning best practices - SWA, AdamW, Ranger optimizer, OneCycle, and more.
- Developments best practices - fp16 support, distributed training, slurm support.


Tests
~~~~~~~~~~~~~~~~~~~~~~
All Catalyst code, features and pipelines `are fully tested`_
with our own `catalyst-codestyle`_.

In fact, we train a number of different models for various of tasks -
image classification, image segmentation, text classification, GANs training
and much more.
During the tests, we compare their convergence metrics in order to verify
the correctness of the training procedure and its reproducibility.

As a result, Catalyst provides fully tested and reproducible
best practices for your deep learning research.

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
    getting_started/migrating_from_other
    Catalyst 101 — Accelerated PyTorch <https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92?source=friends_link&sk=d3dd9b2b23500eca046361187b4619ff>
    Catalyst 102 — Core Trinity <https://medium.com/pytorch/catalyst-102-core-trinity-experiment-runner-and-callback-54adc384b57c?source=friends_link&sk=2aff824412e2f653587a30cd853b030c>

.. toctree::
    :caption: Tutorials
    :maxdepth: 2
    :hidden:

    tutorials/ddp

.. toctree::
    :caption: Core
    :maxdepth: 2
    :hidden:

    core/runner
    core/engine
    core/callback

.. toctree::
    :caption: FAQ
    :maxdepth: 2
    :hidden:

    faq/intro

    faq/data
    faq/lr_finder

    faq/dp
    faq/amp
    faq/ddp
    faq/slurm
    faq/tpu

    faq/multi_components
    faq/early_stopping
    faq/checkpointing
    faq/debugging
    faq/logging
    faq/inference
    faq/finetuning

    faq/stages
    faq/config_api
    faq/optuna


.. toctree::
    :caption: Contribution guide
    :maxdepth: 2
    :hidden:

    How to start <https://github.com/catalyst-team/catalyst/blob/master/CONTRIBUTING.md>
    Codestyle <https://github.com/catalyst-team/codestyle>
    Acknowledgments <https://github.com/catalyst-team/catalyst#acknowledgments>


.. toctree::
    :caption: API

    api/callbacks
    api/contrib
    api/core
    api/data


