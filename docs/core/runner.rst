Runner
==============================================================================

``Runner`` is an abstraction that takes all the logic of your deep learning experiment:
the data you are using, the model you are training,
the batch handling logic, and everything about the selected metrics and monitoring systems:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst21-post-pics/main/code_runner21.png
    :alt: Runner


The ``Runner`` has the most crucial role
in connecting all other abstractions and defining the whole experiment logic into one place.
Most importantly, it does not force you to use Catalyst-only primitives.
It gives you a flexible way to determine
the level of high-level API you want to get from the framework.

For example, you could:

- Define everything in a Catalyst-way with Runner and Callbacks: `ML — multiclass classification`_ example.
- Write forward-backward on your own, using Catalyst as a for-loop wrapper: `custom batch-metrics logging`_ example.
- Mix these approaches: `CV — MNIST GAN`_, `CV — MNIST VAE`_ examples.

Finally, the ``Runner`` architecture does not depend on PyTorch in any case, providing directions for adoption for Tensorflow2 or JAX support. 
Supported Runners are listed under the `Runner API section`_.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`ML — multiclass classification`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`custom batch-metrics logging`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`CV — MNIST GAN`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`CV — MNIST VAE`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`Runner API section`: https://catalyst-team.github.io/catalyst/api/runners.html
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
