Runner
==============================================================================

Runner — an abstraction that knows **how** to run an experiment.
It contains all the logic of how to work with your model per-
stage, epoch, and batch.

From my experience, deep learning experiments follow the same for-loop
(how to run):

.. code-block:: python

    for stage in experiment.stages:
        for epoch in stage.epochs:
            for loader in epoch.loaders:
                for batch in loader:
                    handle_batch(batch)

The only thing that we want to change in these pipelines
for new data/model is batch-handler.
This is exactly what our Runner is doing –
it goes through stages and runs a common train-loop +
extra tricks for reproducibility.

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/catalyst102-runner.png
    :alt: Runner

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw