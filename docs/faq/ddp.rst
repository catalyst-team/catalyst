Distributed training
==============================================================================
Catalyst supports automatic experiments scaling with distributed training support.

Notebook API
----------------------------------------------------

Suppose you have the following pipeline with Linear Regression:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst.dl import SupervisedRunner

    # experiment setup
    logdir = "./logdir"
    num_epochs = 8

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
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
    )

For correct DDP training, you need to separate creation of the dataset(s) from the training.
In this way Catalyst could easily transfer your datasets to the distributed mode
and there would be no data re-creation on each worker.

As a best practice scenario for this case:

.. code-block:: python

    import torch
    from torch.utils.data import TensorDataset
    from catalyst.dl import SupervisedRunner, utils

    def datasets_fn(num_features: int):
        X = torch.rand(int(1e4), num_features)
        y = torch.rand(X.shape[0])
        dataset = TensorDataset(X, y)
        return {"train": dataset, "valid": dataset}

    def train():
        num_features = int(1e1)
        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        runner = SupervisedRunner()
        runner.train(
            model=model,
            datasets={
                "batch_size": 32,
                "num_workers": 1,
                "get_datasets_fn": datasets_fn,
                "num_features": num_features,
            },
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            logdir="./logs/example_3",
            num_epochs=8,
            verbose=True,
            distributed=False,
        )

    utils.distributed_cmd_run(train)

Config API
----------------------------------------------------
To run Catalyst experiments in the DDP-mode,
the only thing you need to do for the Config API - pass required flag to the ``run`` command:

.. code-block:: bash

    catalyst-dl run -C=/path/to/configs --distributed

Launch your training
----------------------------------------------------

In your terminal,
type the following line (adapt `script_name` to your script name ending with .py).

.. code-block:: bash

    python {script_name}

You can vary availble GPUs with ``CUDA_VIBIBLE_DEVICES`` option, for example,

.. code-block:: bash

    # run only on 1st and 2nd GPUs
    CUDA_VISIBLE_DEVICES="1,2" python {script_name}

.. code-block:: bash

    # run only on 0, 1st and 3rd GPUs
    CUDA_VISIBLE_DEVICES="0,1,3" python {script_name}


What will happen is that the same model will be copied on all your available GPUs.
During training, the full dataset will randomly split between the GPUs
(that will change at each epoch).
Each GPU will grab a batch (on that fraction of the dataset),
pass it through the model, compute the loss then back-propagate (to calculate the gradients).
Then they will share their results and average them,
which means like your training is the equivalent of a training
with a batch size of ```batch_size x num_gpus``
(where ``batch_size`` is what you used in your script).

Since they all have the same gradients at this stage,
they will all perform the same update,
so the models will still be the same after this step.
Then training continues with the next batch,
until the number of desired iterations is done.

During training Catalyst will automatically average all metrics
and log them on ``Master`` node only. Same logic used for model checkpointing.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
