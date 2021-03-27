Mixed precision training
==============================================================================
Catalyst support a variety of backends for mixed precision training.
For the PyTorch versions below 1.6 it's better to use ``Nvidia Apex`` helper.
After PyTorch 1.6 release, it's possible to use AMP natively inside ``torch.amp`` package.

Suppose you have the following pipeline with Linear Regression:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst.dl import SupervisedRunner

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
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=8,
        verbose=True,
    )

Nvidia Apex
----------------------------------------------------
To use Nvidia Apex fp16 support you firstly need to install it with,

.. code-block:: bash

    !git clone https://github.com/NVIDIA/apex
    !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

After that you could easily extend our current pipeline with just one line of code:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst.dl import SupervisedRunner

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
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=8,
        verbose=True,
        apex=True,
    )

You could also check out the example above in `this Google Colab notebook`_

Torch AMP
----------------------------------------------------
If you would like to use native AMP support, you could do the following:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst.dl import SupervisedRunner

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
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=8,
        verbose=True,
        amp=True,
    )

You could also check out the example above in `this Google Colab notebook`_

.. _`this Google Colab notebook`: https://colab.research.google.com/drive/12ONaj4sMPiOT_64wh2bpH_AvRCuNFxLx?usp=sharing

Nvidia Apex (Config API)
----------------------------------------------------

Firstly, prepare the config. For example:

.. code-block:: yaml

    engine:
        _target_: APEXEngine
        opt_level: "O1"
        ...

After that just run:

.. code-block:: bash

    catalyst-dl run -C=/path/to/configs --apex

Torch AMP (Config API)
----------------------------------------------------

For native AMP support you only need to pass required flag to the ``run`` command:

.. code-block:: bash

    catalyst-dl run -C=/path/to/configs --amp

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
