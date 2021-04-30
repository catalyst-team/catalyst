Distributed training tutorial
==============================================================================

If you have multiple GPUs,
the most reliable way to use all of them for training is to use the distributed package from pytorch.
To help you, there is a distributed helpers in Catalyst to make it really easy.

Note, that current distributed implementation requires you
to run only training procedure in your python scripts.

Prepare your script
------------------------------------------------

Distributed training doesn't work in a notebook, so prepare a script to run the training.
For instance, here is a minimal script that trains a linear regression model.

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

Stage 1 - 1 line ddp
------------------------------------------------

In case you want to run it fast and ugly, with minimal changes,
you can just pass ``ddp=True`` to ``.train`` call

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
        logdir="./logs/example_1",
        num_epochs=8,
        verbose=True,
        ddp=True,
    )

In this way Catalyst
will try to automatically make your loaders work in distributed setup
and will run experiment training.

Nevertheless it has several disadvantages,
    - you create your loader again and again with each distributed worker,
      +1 for master scripts with all processes joined.
    - you can't understand what is going under the hood of ``ddp=True``
    - we can't always transfer your loaders to distributed mode correctly

Case 2 - We are going deeper
------------------------------------------------

Let's make it more reusable:

.. code-block:: python

    import torch
    from torch.utils.data import TensorDataset
    from catalyst.dl import SupervisedRunner

    def datasets_fn(num_features: int):
        X = torch.rand(int(1e4), num_features)
        y = torch.rand(X.shape[0])
        dataset = TensorDataset(X, y)
        return {"train": dataset, "valid": dataset}

    num_features = int(1e1)
    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders={
            "batch_size": 32,
            "num_workers": 1,
            "datasets_fn": datasets_fn,
            "num_features": num_features,
        },
        logdir="./logs/example_2",
        num_epochs=8,
        verbose=True,
        ddp=True,
    )

Advantages,
    - you don't duplicate the data - it calls when it really needed
    - we still can easily transfer them to distributed mode,
      thanks to ``Datasets`` usage

Disadvantages,
    - everything works thanks to Catalyst `internal logic`_,
      which could be complicated to understand

Could we make everything precisely clear?

.. _`internal logic`: https://github.com/catalyst-team/catalyst/blob/master/catalyst/runners/runner.py#L147

Case 3 - Best practices for distributed training
------------------------------------------------

Yup, check this one, distributed training like a pro:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from catalyst import dl, utils

    num_features = int(1e1)
    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    class CustomSupervisedRunner(dl.SupervisedRunner):
        def get_loaders(self, stage: str):
            X = torch.rand(int(1e4), num_features)
            y = torch.rand(X.shape[0])
            is_ddp = utils.get_rank() > -1

            dataset = TensorDataset(X, y)
            sampler = DistributedSampler(dataset) if is_ddp else None
            loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=32, num_workers=1)

            return {"train": dataset, "valid": dataset}

    runner = CustomSupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=None,  # as far as we have rewrite the loader logic already
        logdir="./logs/example_3",
        num_epochs=8,
        verbose=True,
        ddp=True,
    )

Advantages,
    - you don't duplicate the data - it calls when it really needed
    - we still can easily transfer them to distributed mode, thanks to ``Datasets`` usage
    - the code is easily readable thanks to pure PyTorch way of working with datasets

Launch your training
------------------------------------------------

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
During training, the full dataset will randomly be split between the GPUs
(that will change at each epoch).
Each GPU will grab a batch (on that fractioned dataset),
pass it through the model, compute the loss then back-propagate the gradients.
Then they will share their results and average them,
which means like your training is the equivalent of a training
with a batch size of ```batch_size x num_gpus``
(where ``batch_size`` is what you used in your script).

Since they all have the same gradients at this stage,
they will al perform the same update,
so the models will still be the same after this step.
Then training continues with the next batch,
until the number of desired iterations is done.

During training Catalyst will automatically average all metrics
and log them on ``Master`` node only. Same logic used for model checkpointing.


Slurm support
------------------------------------------------

Catalyst supports distributed training of neural networks on HPC under slurm control.
Catalyst automatically allocates roles between nodes and syncs them.
This allows to run experiments without any changes in the configuration file or model code.
We recommend using nodes with the same number and type of GPU.
You can run the experiment with the following command:

.. code-block:: bash

    # Catalyst Notebook API
    srun -N 2 --gres=gpu:3 --exclusive --mem=256G python run.py
    # Catalyst Config API
    srun -N 2 --gres=gpu:3 --exclusive --mem=256G catalyst-dl run -C config.yml


In this command,
we request two nodes with 3 GPUs on each node in exclusive mode,
i.e. we request all available CPUs on the nodes.
Each node will be allocated 256G.
Note that specific startup parameters using ``srun``
may change depending on the specific cluster and slurm settings.
For more fine-tuning, we recommend reading the slurm documentation.
