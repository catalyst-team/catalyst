Settings
==============================================================================

Catalyst extensions
----------------------------------------------------

Catalyst has a wide variety of framework extensions availabe if you need them.
The base Catalyst package made as light as possible to extend only the PyTorch.
Nevertheless, there are much more availabe:

.. code-block:: bash

    pip install catalyst[comet] # + comet_ml
    pip install catalyst[cv] # + imageio, opencv, scikit-image, torchvision, Pillow
    pip install catalyst[deepspeed] # + deepspeed
    pip install catalyst[dev] # used for Catalyst development and documentaiton rendering
    pip install catalyst[ml] # + scipy, matplotlib, pandas, scikit-learn
    pip install catalyst[mlflow] # + mlflow
    pip install catalyst[neptune] # + neptune-client
    pip install catalyst[onnx-gpu] # + onnx, onnxruntime-gpu
    pip install catalyst[onnx] # + onnx, onnxruntime
    pip install catalyst[optuna] # + optuna
    pip install catalyst[profiler] # + profiler
    pip install catalyst[wandb] # + wandb
    pip install catalyst[all] # + catalyst[cv], catalyst[ml], catalyst[optuna]


As far as you can see, Catalyst has a lot of extensions, and not all of them are strictly required for everyday PyTorch use, so they are extras.
Please see the documentaiton for notes about extra requirements.


Settigns
----------------------------------------------------

To make your workflow reproducible, you could create a ``.catalyst`` file under your project or root directory so that Catalyst could understand all needed requirements during framework initialization.
For example:


.. code-block:: bash

    [catalyst]
    cv_required = false
    mlflow_required = false
    ml_required = true
    neptune_required = false
    optuna_required = false

With such a configuration file, Catalyst will raise you an error if there are now required `catalyst[ml]` dependencies were found.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
