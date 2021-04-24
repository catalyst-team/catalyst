Logging
==============================================================================

Metrics logging
----------------------------------------------------
Catalyst supports a variety of metrics storages during the experiment

- ``runner.batch_metrics`` - dictionary, flatten storage for batch metrics.
    ::

        runner.batch_metrics = {"loss": ..., "accuracy": ..., "iou": ...}

- ``runner.loader_metrics`` - dictionary with aggregated batch statistics for loader (mean over all batches) and global loader metrics, like AUC.
    ::

        runner.loader_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

- ``runner.epoch_metrics`` - dictionary with summarized metrics for different loaders and global epoch metrics, like lr, momentum.
    ::

        runner.epoch_metrics = {
            "train": {"loss": ..., "accuracy": ..., "auc": ...},
            "valid: {"loss": ..., "accuracy": ..., "auc": ...}
            "_epoch_": {"lr": ..., "momentum": ...,}
        }

You could log any new metric in a straightforward way:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics
    from catalyst.data.transforms import ToTensor
    from catalyst.contrib.datasets import MNIST

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
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

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device))

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
            loss = F.cross_entropy(logits, y)
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
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=5,
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
    )

[WIP] Metrics logging with callbacks
----------------------------------------------------

- todo

[WIP] Supported loggers
----------------------------------------------------

- console
- csv
- Tensorboard
- Mlflow
- Neptune

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
