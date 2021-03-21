Model debugging
==============================================================================

Pipeline debugging
----------------------------------------------------
To check pipeline correctness, that everything is working correctly
and does not throws any error, we suggest to use ``CheckRunCallback``.
You could find more information about it here <../early_stopping.rst>.

To check model convergence withing pipeline,
we suggest to use ``BatchOverfitCallback``.
You could find more information about it here <../data.rst>.

Python debugging
----------------------------------------------------
For python debugging we suggest to use ``ipdb``. You could install it with:

.. code-block:: bash

    pip install ipdb

After that you could stop the pipeline in the place you prefer, for example:

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
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            y_hat = self.model(x.view(x.size(0), -1))

            # let's stop before metric computation, but after model forward pass
            import ipdb; ipdb.set_trace()
            # <--- we will stop here --->
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


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw