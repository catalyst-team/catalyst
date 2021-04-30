Multiple components
==============================================================================

Thanks to Catalyst "key-value is all you need" approach,
it's very easy to use run experiments in multi-components setup
(several model, criterions, optimizers, schedulers).

Suppose you have the following classification pipeline (in pure PyTorch):

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

Multi-model
----------------------------------------------------
Multi-model example:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics, utils
    from catalyst.data.transforms import ToTensor
    from catalyst.contrib.datasets import MNIST

    # <--- multi-model setup --->
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128))
    head = nn.Linear(128, 10)
    model = {"encoder": encoder, "head": head}
    optimizer = optim.Adam([
        {'params': encoder.parameters()},
        {'params': head.parameters()},
    ], lr=0.02)
    # <--- multi-model setup --->
    criterion = nn.CrossEntropyLoss()

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
            # <--- multi-model usage --->
            # run model forward pass
            x_ = self.model["encoder"](x)
            logits = self.model["head"](x_)
            # <--- multi-model usage --->
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

As you can see, the only think you need to do - just wrap the model with key-value.
That it, simple enough, no extra abstractions required.

Multi-optimizer
----------------------------------------------------
Multi-optimizer example:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics, utils
    from catalyst.data.transforms import ToTensor
    from catalyst.contrib.datasets import MNIST

    # <--- multi-model/optimizer setup --->
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128))
    head = nn.Linear(128, 10)
    model = {"encoder": encoder, "head": head}
    optimizer = {
        "encoder": optim.Adam(encoder.parameters(), lr=0.02),
        "head": optim.Adam(head.parameters(), lr=0.001),
    }
    # <--- multi-model/optimizer setup --->
    criterion = nn.CrossEntropyLoss()

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
            # <--- multi-model/optimizer usage --->
            # run model forward pass
            x_ = self.model["encoder"](x)
            logits = self.model["head"](x_)
            # <--- multi-model/optimizer usage --->
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
                # <--- multi-model/optimizer usage --->
                self.optimizer["encoder"].step()
                self.optimizer["head"].step()
                self.optimizer["encoder"].zero_grad()
                self.optimizer["head"].zero_grad()
                # <--- multi-model/optimizer usage --->

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

The same thing here - we could wrap our optimizers with key-value too and use it in a straightforward way.

Multi-criterion
----------------------------------------------------
Multi-criterion example:

.. code-block:: python

    import os
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics, utils
    from catalyst.data.transforms import ToTensor
    from catalyst.contrib.datasets import MNIST

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    # <--- multi-criterion setup --->
    criterion = {
        "multiclass": nn.CrossEntropyLoss(),
        "multilabel": nn.BCEWithLogitsLoss(),
    }
    # <--- multi-criterion setup --->

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
            # <--- multi-criterion usage --->
            # compute the loss
            loss_multiclass = self.criterion["multiclass"](logits, y)
            loss_multilabel = \
                self.criterion["multilabel"](logits, F.one_hot(y, 10).to(torch.float32))
            loss = loss_multiclass + loss_multilabel
            # <--- multi-criterion usage --->
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

Same approach here - just use key-value storage to pass criterion through the experiment.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
