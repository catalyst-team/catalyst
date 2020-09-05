Catalyst
========================================

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


PyTorch framework for Deep Learning research and development.
It focuses on reproducibility, rapid experimentation, and codebase reuse
so you can create something new rather than write another regular train loop.

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
    from catalyst import dl
    from catalyst.utils import metrics

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
4. Read `blogposts`_ with use-cases and guides (and Config API intro).
5. Go through advanced  `classification`_, `detection`_ and `segmentation`_ pipelines with Config API. More pipelines available under `projects section`_.
6. Want more? See `Alchemy`_ and `Reaction`_ packages.
7. For Catalyst.RL introduction, please follow `Catalyst.RL repo`_.

.. _`Catalyst 101 — Accelerated PyTorch`: https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92?source=friends_link&sk=d3dd9b2b23500eca046361187b4619ff
.. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`Notebook Tutorials with Google Colab`: https://github.com/catalyst-team/catalyst#tutorials
.. _`blogposts`: https://github.com/catalyst-team/catalyst#blogposts
.. _`classification: https://github.com/catalyst-team/classification
.. _`detection`: https://github.com/catalyst-team/detection
.. _`segmentation`: https://github.com/catalyst-team/segmentation
.. _`projects section`: https://github.com/catalyst-team/catalyst#projects
.. _Alchemy: https://github.com/catalyst-team/alchemy
.. _Reaction: https://github.com/catalyst-team/reaction
.. _Catalyst.RL repo: https://github.com/catalyst-team/catalyst-rl

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

    pip install catalyst[cv]         # installs CV-based catalyst
    pip install catalyst[nlp]        # installs NLP-based catalyst
    pip install catalyst[ecosystem]  # installs Catalyst.Ecosystem
    # and master version installation
    pip install git+https://github.com/catalyst-team/catalyst@master --upgrade


Catalyst is compatible with: Python 3.6+. PyTorch 1.1+.

Tested on Ubuntu 16.04/18.04/20.04, macOS 10.15, Windows 10 and Windows Subsystem for Linux.


Structure
~~~~~~~~~~~~~~~~~~~~~~
- **core** - framework core with main abstractions - Experiment, Runner and Callback.
- **data** - useful tools and scripts for data processing.
- **dl** – runner for training and inference, all of the classic ML and CV/NLP/RecSys metrics and a variety of callbacks for training, validation and inference of neural networks.
- **tools** - extra tools for Deep Learning research, class-based helpers.
- **utils** - typical utils for Deep Learning research, function-based helpers.
- **contrib** - additional modules contributed by Catalyst users.


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
    :caption: Overview
    :maxdepth: 2
    :hidden:

    self
    info/examples
    info/distributed
    info/contributing


.. toctree::
    :caption: API

    api/core
    api/dl
    api/registry

    api/data
    api/utils
    api/contrib