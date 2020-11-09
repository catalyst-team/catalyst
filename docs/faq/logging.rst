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
            "train_loss": ..., "train_auc": ..., "valid_loss": ...,
            "lr": ..., "momentum": ...,
        }

- ``runner.valid_metrics`` - dictionary with validation metrics for current epoch.
    ::

        runner.valid_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

- ``runner.best_valid_metrics`` - dictionary with best validation metrics during whole training process.
    ::

        runner.best_valid_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

You could log any new metric in a stratforward way:

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics
    from catalyst.data.cv import ToTensor
    from catalyst.contrib.datasets import MNIST

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
            # <--- logging --->
            # here we are adding loss, accuracy01 and accuracy03 to the batch metrics
            self.batch_metrics.update(
                {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
            )
            # <--- logging --->

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

[WIP] Metrics logging with callback
----------------------------------------------------

- todo

[WIP] Supported loggers
----------------------------------------------------

- console
- txt
- Tensorboard
- Alchemy
- Neptune
- Weights and Biases

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw