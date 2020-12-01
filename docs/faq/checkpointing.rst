Model checkpointing
==============================================================================

Experiment checkpoints
----------------------------------------------------
With the help of ``CheckpointCallback``
Catalyst creates the following checkpoints structure under selected ``logdir``:

.. code-block:: bash

    logdir/
        code/ <-- code of your experiment and dump of the catalyst, for reproducibility -->
        checkpoints/ <-- theme of the topic -->
            {stage_name}.{epoch_index}.pth <-- topK checkpoints based on model selection logic -->
            best.pth <-- best model based on specified model selection logic -->
            last.pth <-- last model checkpoint in the whole experiment run -->
            <-- the same checkpoints with ``_full`` prefix -->
        ...

These checkpoints are pure PyTorch checkpoints without any mixins with the following structure:

.. code-block:: bash

    checkpoint.pth = {
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

Full checkpoints
----------------------------------------------------
Catalyst saves 2 types of checkpoints:

- ``{checkpoint}.pth`` - stores only model state dict and could be easily used for deploying in the production.
- ``{checkpoint}_full.pth`` - stores all state dicts for model(s), criterion(s), optimizer(s) and scheduler(s) and could be used for experiment analysis purposes.

Save model
----------------------------------------------------
Catalyst has a user-friendly utils to save the model:

.. code-block:: python

    from catalyst import utils

    model = Net()
    checkpoint = utils.pack_checkpoint(model=model)
    utils.save_checkpoint(checkpoint, logdir="/path/to/logdir", suffix="my_checkpoint")
    #  now you could find your checkpoint under "/path/to/logdir/my_checkpoint.pth" location

Load model
----------------------------------------------------
With Catalyst utils it's very easy to load models after experiment run:

.. code-block:: python

    from catalyst import utils

    model = Net()
    optimizer = ...
    criterion = ...
    checkpoint = utils.load_checkpoint(path="/path/to/checkpoint")
    utils.unpack_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

In this case Catalyst would try to unpack requested state dicts from the checkpoint.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
