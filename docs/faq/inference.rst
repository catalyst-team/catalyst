Inference
==============================================================================

To use your model in the inference mode,
you could redefine the ``Runner.predict_batch``.

Suppose you have the following classification pipeline:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics, utils
    from catalyst.data.transforms import ToTensor
    from catalyst.contrib.datasets import MNIST

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
        ),
        "valid": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
        ),
    }

    class CustomRunner(dl.Runner):

        # <--- model inference step --->
        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device))
        # <--- model inference step --->

        def on_loader_start(self, runner):
            super().on_loader_start(runner)
            self.meters = {
                key: metrics.AdditiveValueMetric(compute_on_call=False)
                for key in ["loss", "accuracy01", "accuracy03"]
            }

        def handle_batch(self, batch):
            # model train/valid step
            # unpack the batch
            x, y = batch
            # run model forward pass
            logits = self.model(x)
            # compute the loss
            loss = self.criterion(logits, y)
            # compute other metrics of interest
            accuracy01, accuracy03 = metrics.accuracy(logits, y, topk=(1, 3))
            # log metrics
            self.batch_metrics.update(
                {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
            )
            for key in ["loss", "accuracy01", "accuracy03"]:
                self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
            # run model backward pass
            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        def on_loader_end(self, runner):
            for key in ["loss", "accuracy01", "accuracy03"]:
                self.loader_metrics[key] = self.meters[key].compute()[0]
            super().on_loader_end(runner)

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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
    )

Now you could easily predict your data with the Runner-specified logic.

Predict batch
----------------------------------------------------
If you want to predict one batch:

.. code-block:: python

    batch_prediciton = runner.predict_batch(next(iter(loaders["valid"])))
    # which would be the same with
    batch_model_prediciton = model(next(iter(loaders["valid"]))[0])
    batch_prediciton == batch_model_prediciton
    >>> True

You could also check out the example above in `this Google Colab notebook`_.

Predict loader
----------------------------------------------------
If you want to predict entire loader:

.. code-block:: python

    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10

The ``runner.predict_loader`` method just iteratively goes through the loader batches,
makes model predictions and yields the results.

You could also check out the example above in `this Google Colab notebook`_.

.. _`this Google Colab notebook`: https://colab.research.google.com/drive/1A_JVXdnecanWCM74qi-KqUn0boElvISk?usp=sharing

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw