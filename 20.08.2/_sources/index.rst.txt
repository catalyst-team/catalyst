Catalyst
========================================

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png
    :target: https://github.com/catalyst-team/catalyst
    :alt: Catalyst logo


PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop.

Break the cycle - use the Catalyst_!

Project manifest_. Part of `PyTorch Ecosystem`_. Part of `Catalyst Ecosystem`_:
    - Alchemy_ - Experiments logging & visualization
    - Catalyst_ - Accelerated DL R&D
    - Reaction_ - Convenient DL serving

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

- `Customizing what happens in train`_
- `Colab with ML, CV, NLP, GANs and RecSys demos`_
- For Catalyst.RL introduction, please follow `Catalyst.RL repo`_.

.. _`Customizing what happens in train`: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/customizing_what_happens_in_train.ipynb
.. _Colab with ML, CV, NLP, GANs and RecSys demos: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb
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


Features
~~~~~~~~~~~~~~~~~~~~~~
- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – all source code and environment variables will be saved.
- Callbacks – reusable train/inference pipeline parts with easy customization.
- Training stages support.
- Deep Learning best practices - SWA, AdamW, Ranger optimizer, OneCycle, and more.
- Developments best practices - fp16 support, distributed training, slurm support.


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


Tutorials
~~~~~~~~~~~~~~~~~~~~~~

- `Demo with minimal examples`_ for ML, CV, NLP, GANs and RecSys
- Detailed `classification tutorial`_
- Advanced `segmentation tutorial`_
- Comprehensive `classification pipeline`_
- Binary and semantic `segmentation pipeline`_
- `Beyond fashion - Deep Learning with Catalyst (Config API)`_
- `Tutorial from Notebook API to Config API (RU)`_

.. _Demo with minimal examples: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb
.. _`classification tutorial`: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
.. _`segmentation tutorial`: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb
.. _`classification pipeline`: https://github.com/catalyst-team/classification
.. _`segmentation pipeline`: https://github.com/catalyst-team/segmentation
.. _`Beyond fashion - Deep Learning with Catalyst (Config API)`: https://evilmartians.com/chronicles/beyond-fashion-deep-learning-with-catalyst
.. _`Tutorial from Notebook API to Config API (RU)`: https://github.com/Bekovmi/Segmentation_tutorial

In the examples_ of the repository, you can find advanced tutorials and Catalyst best practices.

.. _examples: https://github.com/catalyst-team/catalyst/tree/master/examples


Community
----------------------------------------

Contribution guide
~~~~~~~~~~~~~~~~~~~~~~

We appreciate all contributions.
If you are planning to contribute back bug-fixes,
please do so without any further discussion.
If you plan to contribute new features, utility functions or extensions,
please first open an issue and discuss the feature with us.

Please see the `contribution guide`_ for more information.

.. _`contribution guide`: https://github.com/catalyst-team/catalyst/blob/master/CONTRIBUTING.md

By participating in this project, you agree to abide by its `Code of Conduct`_.

.. _`Code of Conduct`: https://github.com/catalyst-team/catalyst/blob/master/CODE_OF_CONDUCT.md

User feedback
~~~~~~~~~~~~~~~~~~~~~~

We have created ``catalyst.team.core@gmail.com`` for "user feedback".
    - If you like the project and want to say thanks, this the right place.
    - If you would like to start a collaboration between your team and Catalyst team to do better Deep Learning R&D - you are always welcome.
    - If you just don't like Github issues and this ways suits you better - feel free to email us.
    - Finally, if you do not like something, please, share it with us and we can see how to improve it.

We appreciate any type of feedback. Thank you!


Citation
~~~~~~~~~~~~~~~~~~~~~~

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