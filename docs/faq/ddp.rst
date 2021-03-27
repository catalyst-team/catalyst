Distributed training
==============================================================================
Catalyst supports automatic experiments scaling with distributed training support.

Notebook API
----------------------------------------------------

Suppose you have the following pipeline with MNIST Classification:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl
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

    runner = dl.SupervisedRunner()
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
            dl.AUCCallback(input_key="logits", target_key="targets"),
            # catalyst[ml] required
            # dl.ConfusionMatrixCallback(
            #     input_key="logits", target_key="targets", num_classes=num_classes
            # ),
        ]
    )


For correct DDP training, you need to separate creation of the dataset(s) from the training.
In this way Catalyst could easily transfer your datasets to the distributed mode
and there would be no data re-creation on each worker.

As a best practice scenario for this case:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    from catalyst import dl

    class CustomRunner(dl.SupervisedRunner):

        def get_engine(self):
            return dl.DistributedDataParallelEngine()

        def get_loaders(self, stage: str):
            train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
            valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
            loaders = {
                "train": DataLoader(
                    train_data, sampler=DistributedSampler(dataset=train_data), batch_size=32
                ),
                "valid": DataLoader(
                    valid_data, sampler=DistributedSampler(dataset=valid_data), batch_size=32
                ),
            }
            return loaders

    # model, criterion, optimizer, scheduler
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner = CustomRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        num_epochs=8,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
            dl.AUCCallback(input_key="logits", target_key="targets"),
            # catalyst[ml] required
            # dl.ConfusionMatrixCallback(
            #     input_key="logits", target_key="targets", num_classes=num_classes
            # ),
        ]
    )

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
