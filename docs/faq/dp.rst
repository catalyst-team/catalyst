DataParallel training (cpu, single/multi-gpu)
==============================================================================
By design, Catalyst tries to use all available GPUs on your machine.
Nevertheless, thanks to Nvidia CUDA design,
it's easy to control GPUs visibility with ``CUDA_VISIBLE_DEVICES`` flag.

CPU training
----------------------------------------------------
If you don't want to use GPUs at all you could set ``CUDA_VISIBLE_DEVICES=""``.

For Notebook API case, do the following **before** your experiment code:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


For Config API case, it is a bit easier:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES="" catalyst-dl run -C=/path/to/configs

Single GPU training
----------------------------------------------------
If you would like to use only one specific GPU during your experiments...

For Notebook API case, do the following **before** your experiment code:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # or "1", "2" - index of the GPU

The same case for Config API:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES="0" catalyst-dl run -C=/path/to/configs

Multi GPU training
----------------------------------------------------
Multi GPU case is quite similar with Single GPU one.

For Notebook API case, do the following **before** your experiment code:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # or "1,2,3"

The same case for Config API:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES="0,1" catalyst-dl run -C=/path/to/configs

Nvidia SMI
----------------------------------------------------
Rather than use GPU indexing, you could also pass their ``UUID`` to the ``CUDA_VISIBLE_DEVICES``.
To list them, do the following (with example output from my server):

.. code-block:: bash

    nvidia-smi -L
    >>> GPU 0: GeForce GTX 1080 Ti (UUID: GPU-62b307fa-ef1b-c0a8-0bb4-7311cce714a8)
    >>> GPU 1: GeForce GTX 1080 Ti (UUID: GPU-2c0d0e85-119e-a260-aed1-49071fc502bc)
    >>> GPU 2: GeForce GTX 1080 Ti (UUID: GPU-7269b4ac-2190-762c-dc34-fa144b1751f9)
    # Here we could see GPU indices and their UUIDs.

With this info, it's also valid to specify GPU by their UUIDs.

For Notebook API:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-62b307fa-ef1b-c0a8-0bb4-7311cce714a8"

For Config API:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES="GPU-62b307fa-ef1b-c0a8-0bb4-7311cce714a8" \
        catalyst-dl run -C=/path/to/configs

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
