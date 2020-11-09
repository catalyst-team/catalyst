Experiment
==============================================================================

Experiment - an abstraction that contains information about the experiment
- a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
It also has information about the data and transformations used.
The Experiment knows **what** you would like to run.

Each deep learning project has several main components.
These primitives define what we want to use during the experiment:

- the data
- the model(s)
- the optimizer(s)
- the loss(es)
- and the scheduler(s) if we need them.

That are the abstractions that Experiment covers in Catalyst,
with a few modifications for easier experiment monitoring
and hyperparameters logging. For each stage of our experiment,
the Experiment provides interfaces to all primitives above + the callbacks.

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/catalyst102-experiment.png
    :alt: Experiment


---

- what is the difference between different Experiments?