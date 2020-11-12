TPU training
==============================================================================
Catalyst support TPU backend for your deep learning experiments.

To use TPU support you firstly need to install it with,

.. code-block:: bash

    VERSION = "1.5"  # possible version: ["1.5" , "20200325", "nightly"]
    !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null
    !python pytorch-xla-env-setup.py --version $VERSION > /dev/null


Notebook API
----------------------------------------------------
Because of the PyTorch - TPU design,
you need to specify the running device and transfer your experiment components on it **before** the experiment run.

Here is a simple example of classification pipeline on TPU:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl, utils

    # sample data
    num_samples, num_features, num_classes = int(1e4), int(1e1), 4
    X = torch.rand(num_samples, num_features)
    y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

    # pytorch loaders
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # device (TPU > GPU > CPU)
    device = utils.get_device()  # <--- TPU device will be selected here
    # we need to specify the device **before** the experiment run.

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # model training
    runner = dl.SupervisedRunner(device=device)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=3,
        callbacks=[dl.AccuracyCallback(num_classes=num_classes)]
    )

You could also check out the example above in `this Google Colab notebook`_

.. _`this Google Colab notebook`: https://colab.research.google.com/drive/1AhvNzTRb3gd3AYhzUfm3dzw8TddlsfhD?usp=sharing

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
