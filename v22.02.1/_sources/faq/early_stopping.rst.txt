Early stopping
==============================================================================

Early stopping
----------------------------------------------------

To use experiment early stopping you could use ``EarlyStoppingCallback``:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    # model training
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=8,
        verbose=True,
        callbacks=[
            dl.EarlyStoppingCallback(
                patience=2, loader_key="valid", metric_key="loss", minimize=True
            )
        ]
    )

Pipeline checking
----------------------------------------------------
You could also check the pipeline
(run only 3 batches per loader, and 3 epochs per stage)
with ``CheckRunCallback``:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    # model training
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=8,
        verbose=True,
        callbacks=[dl.CheckRunCallback(num_batch_steps=3, num_epoch_steps=3)]
    )

You could also use ``runner.train(..., check=True)`` with Notebook API approach:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    # model training
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=8,
        verbose=True,
        check=True,
    )

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw