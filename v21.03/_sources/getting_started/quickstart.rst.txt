Quickstart 101
==============================================================================
**In this quickstart, we’ll show you how to organize your PyTorch code with Catalyst.**

Catalyst goals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- flexibility, keeping the PyTorch simplicity, but removing the boilerplate code.
- readability by decoupling the experiment run.
- reproducibility.
- scalability to any hardware without code changes.
- extensibility for pipeline customization.

Step 1 - Install packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install using `pip package`_:

.. code:: bash

   pip install -U catalyst

.. _`pip package`: https://pypi.org/project/catalyst/

Step 2 - Make python imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from catalyst import dl, metrics

Step 3 - Write PyTorch code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's define **what** we are experimenting with:

.. code-block:: python

    model = torch.nn.Linear(28 * 28, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

Step 4 - Accelerate it with Catalyst
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's define **how** we are running the experiment (in pure PyTorch):

.. code-block:: python

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
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

Step 5 - Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's **train**, **evaluate**, and **trace** your model with a few lines of code.

.. code-block:: python

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

PS. Yes, this file is exactly 101 line.