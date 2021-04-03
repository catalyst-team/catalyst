Multiple components
==============================================================================

Thanks to Catalyst "key-value is all you need" approach,
it's very easy to use run experiments in multi-components setup
(several model, criterions, optimizers, schedulers).

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

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
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

Multi-model
----------------------------------------------------
Multi-model example:

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from catalyst import dl, metrics

    # <--- multi-model setup --->
    encoder = torch.nn.Linear(28 * 28, 128)
    head = torch.nn.Linear(128, 10)
    model = {"encoder": encoder, "head": head}
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': head.parameters()},
    ], lr=0.02)
    # <--- multi-model setup --->
    criterion = torch.nn.CrossEntropyLoss()

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            # <--- multi-model usage --->
            x_ = self.model["encoder"](x.view(x.size(0), -1))
            y_hat = self.model["head"](x_)
            # <--- multi-model usage --->

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

As you can see, the only think you need to do - just wrap the model with key-value.
That it, simple enough, no extra abstractions required.

Multi-optimizer
----------------------------------------------------
Multi-optimizer example:

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from catalyst import dl, metrics

    # <--- multi-model/optimizer setup --->
    encoder = torch.nn.Linear(28 * 28, 128)
    head = torch.nn.Linear(128, 10)
    model = {"encoder": encoder, "head": head}
    optimizer = {
        "encoder": torch.optim.Adam(encoder.parameters(), lr=0.02),
        "head": torch.optim.Adam(head.parameters(), lr=0.001),
    }
    # <--- multi-model/optimizer setup --->
    criterion = torch.nn.CrossEntropyLoss()

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            # <--- multi-model/optimizer usage --->
            x_ = self.model["encoder"](x.view(x.size(0), -1))
            y_hat = self.model["head"](x_)
            # <--- multi-model/optimizer usage --->

            loss = self.criterion(y_hat, y)
            accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
            self.batch_metrics.update(
                {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
            )

            if self.is_train_loader:
                loss.backward()
                # <--- multi-model/optimizer usage --->
                self.optimizer["encoder"].step()
                self.optimizer["head"].step()
                self.optimizer["encoder"].zero_grad()
                self.optimizer["head"].zero_grad()
                # <--- multi-model/optimizer usage --->

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

The same thing here - we could wrap our optimizers with key-value too and use it in a stratforward way.

Multi-criterion
----------------------------------------------------
Multi-criterion example:

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
    # <--- multi-criterion setup --->
    criterion = {
        "multiclass": torch.nn.CrossEntropyLoss(),
        "multilabel": torch.nn.BCEWithLogitsLoss(),
    }
    # <--- multi-criterion setup --->

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            y_hat = self.model(x.view(x.size(0), -1))

            # <--- multi-criterion usage --->
            loss_multiclass = self.criterion["multiclass"](y_hat, y)
            loss_multilabel = self.criterion["multilabel"](y_hat, F.one_hot(y, 10).to(torch.float32))
            loss = loss_multiclass + loss_multilabel
            # <--- multi-criterion usage --->

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

Same approach here - just use key-value storage to pass criterion through the experiment.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
