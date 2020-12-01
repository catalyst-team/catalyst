Slurm training
==============================================================================

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

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
