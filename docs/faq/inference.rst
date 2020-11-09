Inference
==============================================================================

To use your model in the inference mode,
you could redefine the ``Runner.predict_batch``.

Suppose you have the following classification pipeline:

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from catalyst import dl, metrics

    model = torch.nn.Linear(28 * 28, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):

        # <--- model inference step --->
        def predict_batch(self, batch):
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))
        # <--- model inference step --->

        def _handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            y_hat = self.model(x.view(x.size(0), -1))

            loss = self.criterion(y_hat, y)
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
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=5,
        verbose=True,
        load_best_on_end=True,
    )

Now you could easily predict your data with the Runner-specified logic.

Predict batch
----------------------------------------------------
If you want to predict one batch:

.. code-block:: python

    batch_prediciton = runner.predict_batch(next(iter(loaders["valid"])))
    # which would be the same with
    batch_model_prediciton = model(next(iter(loaders["valid"]))[0].view(32, -1))
    batch_prediciton == batch_model_prediciton
    >>> True

Predict loader
----------------------------------------------------
If you want to predict entire loader:

.. code-block:: python

    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10

The ``runner.predict_loader`` method just iteratively goes through the loader batches,
makes model predictions and yields the results.

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw